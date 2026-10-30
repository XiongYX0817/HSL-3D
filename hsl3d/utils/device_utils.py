from __future__ import annotations

from typing import Optional

import torch


def get_device_type() -> str:
    if hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                return "npu"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_module():
    device_type = get_device_type()
    if device_type == "npu":
        return torch.npu
    if device_type == "cuda":
        return torch.cuda
    return None


def get_device_count() -> int:
    device_module = get_device_module()
    if device_module is None:
        return 1
    return device_module.device_count()


def current_device() -> int:
    device_module = get_device_module()
    if device_module is None:
        return 0
    return device_module.current_device()


def set_device(index: int) -> None:
    device_module = get_device_module()
    if device_module is not None:
        device_module.set_device(index)


def manual_seed_all(seed: int) -> None:
    device_module = get_device_module()
    if device_module is not None:
        device_module.manual_seed_all(seed)


def get_torch_device(index: Optional[int] = None) -> torch.device:
    device_type = get_device_type()
    if device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, current_device() if index is None else index)


def get_total_memory_gb(index: Optional[int] = None) -> float:
    device_module = get_device_module()
    if device_module is None:
        return 0.0
    properties = device_module.get_device_properties(current_device() if index is None else index)
    return properties.total_memory / 1024 ** 3


def reset_peak_memory_stats(index: Optional[int] = None) -> None:
    device_module = get_device_module()
    if device_module is not None:
        device_module.reset_peak_memory_stats(current_device() if index is None else index)


def max_memory_allocated_gb(index: Optional[int] = None) -> float:
    device_module = get_device_module()
    if device_module is None:
        return 0.0
    return device_module.max_memory_allocated(current_device() if index is None else index) / 1024 ** 3
