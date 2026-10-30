from abc import abstractmethod
from contextlib import contextmanager
from typing import Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

from .device_utils import current_device, get_total_memory_gb, max_memory_allocated_gb, reset_peak_memory_stats, get_torch_device


class MemoryController:
    """
    Base class for memory management during training.
    """
    
    _last_input_size = None
    _last_mem_ratio = []
    
    @contextmanager
    def record(self):
        pass
    
    def update_run_states(self, input_size=None, mem_ratio=None):
        if self._last_input_size is None:
            self._last_input_size = input_size
        elif self._last_input_size!= input_size:
            raise ValueError(f'Input size should not change for different ElasticModules.')
        self._last_mem_ratio.append(mem_ratio)
    
    @abstractmethod
    def get_mem_ratio(self, input_size):
        pass
    
    @abstractmethod
    def state_dict(self):
        pass
    
    @abstractmethod
    def log(self):
        pass


class LinearMemoryController(MemoryController):
    """
    A simple controller for memory management during training.
    The memory usage is modeled as a linear function of:
        - the number of input parameters
        - the ratio of memory the model use compared to the maximum usage (with no checkpointing)
    memory_usage = k * input_size * mem_ratio + b
    The controller keeps track of the memory usage and gives the
    expected memory ratio to keep the memory usage under a target
    """
    def __init__(
        self,
        buffer_size=1000,
        update_every=500,
        target_ratio=0.8,
        available_memory=None,
        max_mem_ratio_start=0.1,
        sync_initial_mem_ratio_across_ranks=False,
        sync_initial_mem_ratio_source_rank=0,
        params=None,
        device=None
    ):
        self.buffer_size = buffer_size
        self.update_every = update_every
        self.target_ratio = target_ratio
        self.device = current_device() if device is None else device
        self._torch_device = self.device if isinstance(self.device, torch.device) else get_torch_device(self.device)
        self.available_memory = available_memory or get_total_memory_gb(self.device)
        self.sync_initial_mem_ratio_across_ranks = sync_initial_mem_ratio_across_ranks
        self.sync_initial_mem_ratio_source_rank = sync_initial_mem_ratio_source_rank
                
        self._memory = np.zeros(buffer_size, dtype=np.float32)
        self._input_size = np.zeros(buffer_size, dtype=np.float32)
        self._mem_ratio = np.zeros(buffer_size, dtype=np.float32)
        self._buffer_ptr = 0
        self._buffer_length = 0
        self._params = tuple(params) if params is not None else (0.0, 0.0)
        self._max_mem_ratio = max_mem_ratio_start
        self.step = 0

    def __repr__(self):
        return (
            "LinearMemoryController("
            f"target_ratio={self.target_ratio}, "
            f"available_memory={self.available_memory}, "
            f"sync_initial_mem_ratio_across_ranks={self.sync_initial_mem_ratio_across_ranks}"
            ")"
        )

    def _sync_initial_mem_ratio(self, mem_ratio: float) -> float:
        if (
            not self.sync_initial_mem_ratio_across_ranks
            or not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() <= 1
        ):
            return mem_ratio

        sync_tensor = torch.tensor(float(mem_ratio), dtype=torch.float32, device=self._torch_device)
        if dist.get_rank() != self.sync_initial_mem_ratio_source_rank:
            sync_tensor.zero_()
        dist.broadcast(sync_tensor, src=self.sync_initial_mem_ratio_source_rank)
        return float(sync_tensor.item())
        
    def _add_sample(self, memory, input_size, mem_ratio):
        self._memory[self._buffer_ptr] = memory
        self._input_size[self._buffer_ptr] = input_size
        self._mem_ratio[self._buffer_ptr] = mem_ratio
        self._buffer_ptr = (self._buffer_ptr + 1) % self.buffer_size
        self._buffer_length = min(self._buffer_length + 1, self.buffer_size)
            
    @contextmanager
    def record(self):
        reset_peak_memory_stats(self.device)
        self._last_input_size = None
        self._last_mem_ratio = []
        yield
        self._last_memory = max_memory_allocated_gb(self.device)
        self._last_mem_ratio = sum(self._last_mem_ratio) / len(self._last_mem_ratio)
        self._add_sample(self._last_memory, self._last_input_size, self._last_mem_ratio)
        self.step += 1
        if self.step % self.update_every == 0:
            self._max_mem_ratio = min(1.0, self._max_mem_ratio + 0.1)
            self._fit_params()
            
    def _fit_params(self):
        memory_usage = self._memory[:self._buffer_length]
        input_size = self._input_size[:self._buffer_length]
        mem_ratio = self._mem_ratio[:self._buffer_length]
        
        x = input_size * mem_ratio
        y = memory_usage
        k, b = np.polyfit(x, y, 1)
        self._params = (k, b)
        # self._visualize()
        
    def _visualize(self):
        import matplotlib.pyplot as plt
        memory_usage = self._memory[:self._buffer_length]
        input_size = self._input_size[:self._buffer_length]
        mem_ratio = self._mem_ratio[:self._buffer_length]
        k, b = self._params
        
        plt.scatter(input_size * mem_ratio, memory_usage, c=mem_ratio, cmap='viridis')
        x = np.array([0.0, 20000.0])
        plt.plot(x, k * x + b, c='r')
        plt.savefig(f'linear_memory_controller_{self.step}.png')
        plt.cla()
        
    def get_mem_ratio(self, input_size):
        k, b = self._params
        if k == 0:
            mem_ratio = np.random.rand() * self._max_mem_ratio
            return self._sync_initial_mem_ratio(mem_ratio)
        pred = (self.available_memory * self.target_ratio - b) / (k * input_size)
        return min(self._max_mem_ratio, max(0.0, pred))
    
    def state_dict(self):
        return {
            'params': self._params,
        }
        
    def load_state_dict(self, state_dict):
        self._params = tuple(state_dict['params'])
        
    def log(self):
        return {
            'params/k': self._params[0],
            'params/b': self._params[1],
            'memory': self._last_memory,
            'input_size': self._last_input_size,
            'mem_ratio': self._last_mem_ratio,
        }
    
    
class ElasticModule(nn.Module):
    """
    Module for training with elastic memory management.
    """
    def __init__(self):
        super().__init__()
        self._memory_controller: MemoryController = None
        
    @abstractmethod
    def _get_input_size(self, *args, **kwargs) -> int:
        """
        Get the size of the input data.
        
        Returns:
            int: The size of the input data.
        """
        pass
    
    @abstractmethod
    def _forward_with_mem_ratio(self, *args, mem_ratio=0.0, **kwargs) -> Tuple[float, Tuple]:
        """
        Forward with a given memory ratio.
        """
        pass
    
    def register_memory_controller(self, memory_controller: MemoryController):
        self._memory_controller = memory_controller
        
    def forward(self, *args, **kwargs):
        if self._memory_controller is None or not torch.is_grad_enabled() or not self.training:
            _, ret = self._forward_with_mem_ratio(*args, **kwargs)
        else:
            input_size = self._get_input_size(*args, **kwargs)
            mem_ratio = self._memory_controller.get_mem_ratio(input_size)
            mem_ratio, ret = self._forward_with_mem_ratio(*args, mem_ratio=mem_ratio, **kwargs)
            self._memory_controller.update_run_states(input_size, mem_ratio)
        return ret
    

class ElasticModuleMixin:
    """
    Mixin for training with elastic memory management.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory_controller: MemoryController = None
        
    @abstractmethod
    def _get_input_size(self, *args, **kwargs) -> int:
        """
        Get the size of the input data.
        
        Returns:
            int: The size of the input data.
        """
        pass
    
    @abstractmethod
    @contextmanager
    def with_mem_ratio(self, mem_ratio=1.0) -> float:
        """
        Context manager for training with a reduced memory ratio compared to the full memory usage.
        
        Returns:
            float: The exact memory ratio used during the forward pass.
        """
        pass
    
    def register_memory_controller(self, memory_controller: MemoryController):
        self._memory_controller = memory_controller
        
    def forward(self, *args, **kwargs):
        if self._memory_controller is None or not torch.is_grad_enabled() or not self.training:
            ret = super().forward(*args, **kwargs)
        else:
            input_size = self._get_input_size(*args, **kwargs)
            mem_ratio = self._memory_controller.get_mem_ratio(input_size)
            with self.with_mem_ratio(mem_ratio) as exact_mem_ratio:
                ret = super().forward(*args, **kwargs)
            self._memory_controller.update_run_states(input_size, exact_mem_ratio)
        return ret
