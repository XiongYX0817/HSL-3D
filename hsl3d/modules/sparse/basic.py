from typing import *
from fractions import Fraction
import torch
from . import config


__all__ = [
    'VarLenTensor',
    'varlen_cat',
    'varlen_unbind',
    'SparseTensor',
    'sparse_cat',
    'sparse_batchwise_cat',
    'sparse_select_by_rows',
    'sparse_select_by_coords',
    'sparse_unbind',
]


class VarLenTensor:
    """
    Sequential tensor with variable length.
    
    Args:
        feats (torch.Tensor): Features of the varlen tensor.
        layout (List[slice]): Layout of the varlen tensor for each batch
    """
    def __init__(self, feats: torch.Tensor, layout: List[slice]=None):
        self.feats = feats
        self.layout = layout if layout is not None else [slice(0, feats.shape[0])]
        self._cache = {}
        
    @staticmethod
    def layout_from_seqlen(seqlen: list) -> List[slice]:
        """
        Create a layout from a tensor of sequence lengths.
        """
        layout = []
        start = 0
        for l in seqlen:
            layout.append(slice(start, start + l))
            start += l
        return layout
        
    @staticmethod
    def from_tensor_list(tensor_list: List[torch.Tensor]) -> 'VarLenTensor':
        """
        Create a VarLenTensor from a list of tensors.
        """
        feats = torch.cat(tensor_list, dim=0)
        layout = []
        start = 0
        for tensor in tensor_list:
            layout.append(slice(start, start + tensor.shape[0]))
            start += tensor.shape[0]
        return VarLenTensor(feats, layout)
    
    def to_tensor_list(self) -> List[torch.Tensor]:
        """
        Convert a VarLenTensor to a list of tensors.
        """
        tensor_list = []
        for s in self.layout:
            tensor_list.append(self.feats[s])
        return tensor_list
    
    def __len__(self) -> int:
        return len(self.layout)
    
    @property
    def shape(self) -> torch.Size:
        return torch.Size([len(self.layout), *self.feats.shape[1:]])
    
    def dim(self) -> int:
        return len(self.shape)
    
    @property
    def ndim(self) -> int:
        return self.dim()

    @property
    def dtype(self):
        return self.feats.dtype

    @property
    def device(self):
        return self.feats.device
    
    @property
    def seqlen(self) -> torch.LongTensor:
        if 'seqlen' not in self._cache:
            self._cache['seqlen'] = torch.tensor([l.stop - l.start for l in self.layout], dtype=torch.long, device=self.device)
        return self._cache['seqlen']
    
    @property
    def cum_seqlen(self) -> torch.LongTensor:
        if 'cum_seqlen' not in self._cache:
            self._cache['cum_seqlen'] = torch.cat([
                torch.tensor([0], dtype=torch.long, device=self.device),
                self.seqlen.cumsum(dim=0)
            ], dim=0)
        return self._cache['cum_seqlen']
    
    @property
    def batch_boardcast_map(self) -> torch.LongTensor:
        """
        Get the broadcast map for the varlen tensor.
        """
        if 'batch_boardcast_map' not in self._cache:
            self._cache['batch_boardcast_map'] = torch.repeat_interleave(
                torch.arange(len(self.layout), device=self.device),
                self.seqlen,
            )
        return self._cache['batch_boardcast_map']
    
    @overload
    def to(self, dtype: torch.dtype, *, non_blocking: bool = False, copy: bool = False) -> 'VarLenTensor': ...

    @overload
    def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None, *, non_blocking: bool = False, copy: bool = False) -> 'VarLenTensor': ...

    def to(self, *args, **kwargs) -> 'VarLenTensor':
        device = None
        dtype = None
        if len(args) == 2:
            device, dtype = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
        if 'dtype' in kwargs:
            assert dtype is None, "to() received multiple values for argument 'dtype'"
            dtype = kwargs['dtype']
        if 'device' in kwargs:
            assert device is None, "to() received multiple values for argument 'device'"
            device = kwargs['device']
        non_blocking = kwargs.get('non_blocking', False)
        copy = kwargs.get('copy', False)
        
        new_feats = self.feats.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        return self.replace(new_feats)

    def type(self, dtype):
        new_feats = self.feats.type(dtype)
        return self.replace(new_feats)

    def cpu(self) -> 'VarLenTensor':
        new_feats = self.feats.cpu()
        return self.replace(new_feats)
    
    def cuda(self) -> 'VarLenTensor':
        new_feats = self.feats.cuda()
        return self.replace(new_feats)

    def half(self) -> 'VarLenTensor':
        new_feats = self.feats.half()
        return self.replace(new_feats)
    
    def float(self) -> 'VarLenTensor':
        new_feats = self.feats.float()
        return self.replace(new_feats)
    
    def detach(self) -> 'VarLenTensor':
        new_feats = self.feats.detach()
        return self.replace(new_feats)

    def reshape(self, *shape) -> 'VarLenTensor':
        new_feats = self.feats.reshape(self.feats.shape[0], *shape)
        return self.replace(new_feats)
    
    def unbind(self, dim: int) -> List['VarLenTensor']:
        return varlen_unbind(self, dim)

    def replace(self, feats: torch.Tensor) -> 'VarLenTensor':
        new_tensor = VarLenTensor(
            feats=feats,
            layout=self.layout,
        )
        new_tensor._cache = self._cache
        return new_tensor
    
    def to_dense(self, max_length=None) -> torch.Tensor:
        """
        Convert a VarLenTensor to a dense representation without for-loop.
        
        Returns:
            dense (torch.Tensor): (N, L, C) dense tensor
            mask (torch.BoolTensor): (N, L) mask indicating valid positions
        """
        N = len(self)
        L = max_length or self.seqlen.max().item()
        spatial = self.feats.shape[1:]
        idx = torch.arange(L, device=self.device).unsqueeze(0).expand(N, L)
        mask = (idx < self.seqlen.unsqueeze(1))
        mapping = mask.reshape(-1).cumsum(dim=0) - 1
        dense = self.feats[mapping]
        dense = dense.reshape(N, L, *spatial)
        return dense, mask

    def __neg__(self) -> 'VarLenTensor':
        return self.replace(-self.feats)
    
    def __elemwise__(self, other: Union[torch.Tensor, 'VarLenTensor'], op: callable) -> 'VarLenTensor':
        if isinstance(other, torch.Tensor):
            try:
                other = torch.broadcast_to(other, self.shape)
                other = other[self.batch_boardcast_map]
            except:
                pass
        if isinstance(other, VarLenTensor):
            other = other.feats
        new_feats = op(self.feats, other)
        new_tensor = self.replace(new_feats)
        return new_tensor

    def __add__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.add)

    def __radd__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.add)
    
    def __sub__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.sub)
    
    def __rsub__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, lambda x, y: torch.sub(y, x))

    def __mul__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.mul)

    def __rmul__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.mul)

    def __truediv__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.div)

    def __rtruediv__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, lambda x, y: torch.div(y, x))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = range(*idx.indices(self.shape[0]))
        elif isinstance(idx, list):
            assert all(isinstance(i, int) for i in idx), f"Only integer indices are supported: {idx}"
        elif isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                assert idx.shape == (self.shape[0],), f"Invalid index shape: {idx.shape}"
                idx = idx.nonzero().squeeze(1)
            elif idx.dtype in [torch.int32, torch.int64]:
                assert len(idx.shape) == 1, f"Invalid index shape: {idx.shape}"
            else:
                raise ValueError(f"Unknown index type: {idx.dtype}")
        else:
            raise ValueError(f"Unknown index type: {type(idx)}")
        
        new_feats = []
        new_layout = []
        start = 0
        for new_idx, old_idx in enumerate(idx):
            new_feats.append(self.feats[self.layout[old_idx]])
            new_layout.append(slice(start, start + len(new_feats[-1])))
            start += len(new_feats[-1])
        new_feats = torch.cat(new_feats, dim=0).contiguous()
        new_tensor = VarLenTensor(feats=new_feats, layout=new_layout)
        return new_tensor
    
    def reduce(self, op: str, dim: Optional[Union[int, Tuple[int,...]]] = None, keepdim: bool = False) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        
        if op =='mean':
            red = self.feats.mean(dim=dim, keepdim=keepdim)
        elif op =='sum':
            red = self.feats.sum(dim=dim, keepdim=keepdim)
        elif op == 'prod':
            red = self.feats.prod(dim=dim, keepdim=keepdim)
        else:
            raise ValueError(f"Unsupported reduce operation: {op}")
        
        if dim is None or 0 in dim:
            return red
        
        red = torch.segment_reduce(red, reduce=op, lengths=self.seqlen)
        return red
    
    def mean(self, dim: Optional[Union[int, Tuple[int,...]]] = None, keepdim: bool = False) -> torch.Tensor:
        return self.reduce(op='mean', dim=dim, keepdim=keepdim)
        
    def sum(self, dim: Optional[Union[int, Tuple[int,...]]] = None, keepdim: bool = False) -> torch.Tensor:
        return self.reduce(op='sum', dim=dim, keepdim=keepdim)
        
    def prod(self, dim: Optional[Union[int, Tuple[int,...]]] = None, keepdim: bool = False) -> torch.Tensor:
        return self.reduce(op='prod', dim=dim, keepdim=keepdim)
    
    def std(self, dim: Optional[Union[int, Tuple[int,...]]] = None, keepdim: bool = False) -> torch.Tensor:
        mean = self.mean(dim=dim, keepdim=True)
        mean2 = self.replace(self.feats ** 2).mean(dim=dim, keepdim=True)
        std = (mean2 - mean ** 2).sqrt()
        return std
    
    def __repr__(self) -> str:
        return f"VarLenTensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"


def varlen_cat(inputs: List[VarLenTensor], dim: int = 0) -> VarLenTensor:
    """
    Concatenate a list of varlen tensors.
    
    Args:
        inputs (List[VarLenTensor]): List of varlen tensors to concatenate.
    """
    if dim == 0:
        new_feats = torch.cat([input.feats for input in inputs], dim=0)
        start = 0
        new_layout = []
        for input in inputs:
            for l in input.layout:
                new_layout.append(slice(start, start + l.stop - l.start))
                start += l.stop - l.start
        output = VarLenTensor(feats=new_feats, layout=new_layout)
    else:
        feats = torch.cat([input.feats for input in inputs], dim=dim)
        output = inputs[0].replace(feats)

    return output


def varlen_unbind(input: VarLenTensor, dim: int) -> Union[List[VarLenTensor]]:
    """
    Unbind a varlen tensor along a dimension.
    
    Args:
        input (VarLenTensor): Varlen tensor to unbind.
        dim (int): Dimension to unbind.
    """
    if dim == 0:
        return [input[i] for i in range(len(input))]
    else:
        feats = input.feats.unbind(dim)
        return [input.replace(f) for f in feats]
    

class SparseTensor(VarLenTensor):
    """
    Sparse tensor with support for both torchsparse and spconv backends.
    
    Parameters:
    - feats (torch.Tensor): Features of the sparse tensor.
    - coords (torch.Tensor): Coordinates of the sparse tensor.
    - shape (torch.Size): Shape of the sparse tensor.
    - layout (List[slice]): Layout of the sparse tensor for each batch
    - data (SparseTensorData): Sparse tensor data used for convolusion

    NOTE:
    - Data corresponding to a same batch should be contiguous.
    - Coords should be in [0, 1023]
    """
    SparseTensorData = None

    @overload
    def __init__(self, feats: torch.Tensor, coords: torch.Tensor, shape: Optional[torch.Size] = None, **kwargs): ...

    @overload
    def __init__(self, data, shape: Optional[torch.Size] = None, **kwargs): ...

    def __init__(self, *args, **kwargs):
        # Lazy import of sparse tensor backend
        if self.SparseTensorData is None:
            import importlib
            if config.CONV == 'torchsparse':
                self.SparseTensorData = importlib.import_module('torchsparse').SparseTensor
            elif config.CONV == 'spconv':
                self.SparseTensorData = importlib.import_module('spconv.pytorch').SparseConvTensor
                
        method_id = 0
        if len(args) != 0:
            method_id = 0 if isinstance(args[0], torch.Tensor) else 1
        else:
            method_id = 1 if 'data' in kwargs else 0

        if method_id == 0:
            feats, coords, shape = args + (None,) * (3 - len(args))
            if 'feats' in kwargs:
                feats = kwargs['feats']
                del kwargs['feats']
            if 'coords' in kwargs:
                coords = kwargs['coords']
                del kwargs['coords']
            if 'shape' in kwargs:
                shape = kwargs['shape']
                del kwargs['shape']

            if config.CONV == 'torchsparse':
                self.data = self.SparseTensorData(feats, coords, **kwargs)
            elif config.CONV == 'spconv':
                spatial_shape = list(coords.max(0)[0] + 1)
                self.data = self.SparseTensorData(feats.reshape(feats.shape[0], -1), coords, spatial_shape[1:], spatial_shape[0], **kwargs)
                self.data._features = feats
            else:
                self.data = {
                    'feats': feats,
                    'coords': coords,
                }
        elif method_id == 1:
            data, shape = args + (None,) * (2 - len(args))
            if 'data' in kwargs:
                data = kwargs['data']
                del kwargs['data']
            if 'shape' in kwargs:
                shape = kwargs['shape']
                del kwargs['shape']

            self.data = data

        self._shape = shape
        self._scale = kwargs.get('scale', (Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)))
        self._spatial_cache = kwargs.get('spatial_cache', {})
        self._lookup_cache = {}

        if config.DEBUG:
            try:
                assert self.feats.shape[0] == self.coords.shape[0], f"Invalid feats shape: {self.feats.shape}, coords shape: {self.coords.shape}"
                assert self.shape == self.__cal_shape(self.feats, self.coords), f"Invalid shape: {self.shape}"
                assert self.layout == self.__cal_layout(self.coords, self.shape[0]), f"Invalid layout: {self.layout}"
                for i in range(self.shape[0]):
                    assert torch.all(self.coords[self.layout[i], 0] == i), f"The data of batch {i} is not contiguous"
            except Exception as e:
                print('Debugging information:')
                print(f"- Shape: {self.shape}")
                print(f"- Layout: {self.layout}")
                print(f"- Scale: {self._scale}")
                print(f"- Coords: {self.coords}")
                raise e
        
    @staticmethod
    def from_tensor_list(feats_list: List[torch.Tensor], coords_list: List[torch.Tensor]) -> 'SparseTensor':
        """
        Create a SparseTensor from a list of tensors.
        """
        feats = torch.cat(feats_list, dim=0)
        coords = []
        for i, coord in enumerate(coords_list):
            coord = torch.cat([torch.full_like(coord[:, :1], i), coord[:, 1:]], dim=1)
            coords.append(coord)
        coords = torch.cat(coords, dim=0)
        return SparseTensor(feats, coords)
    
    def to_tensor_list(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Convert a SparseTensor to list of tensors.
        """
        feats_list = []
        coords_list = []
        for s in self.layout:
            feats_list.append(self.feats[s])
            coords_list.append(self.coords[s])
        return feats_list, coords_list
    
    def __len__(self) -> int:
        return len(self.layout)
        
    def __cal_shape(self, feats, coords):
        shape = []
        shape.append(coords[:, 0].max().item() + 1)
        shape.extend([*feats.shape[1:]])
        return torch.Size(shape)
    
    def __cal_layout(self, coords, batch_size):
        seq_len = torch.bincount(coords[:, 0], minlength=batch_size)
        offset = torch.cumsum(seq_len, dim=0) 
        layout = [slice((offset[i] - seq_len[i]).item(), offset[i].item()) for i in range(batch_size)]
        return layout
    
    def __cal_spatial_shape(self, coords):
        return torch.Size((coords[:, 1:].max(0)[0] + 1).tolist())
    
    @property
    def shape(self) -> torch.Size:
        if self._shape is None:
            self._shape = self.__cal_shape(self.feats, self.coords)
        return self._shape
    
    @property
    def layout(self) -> List[slice]:
        layout = self.get_spatial_cache('layout')
        if layout is None:
            layout = self.__cal_layout(self.coords, self.shape[0])
            self.register_spatial_cache('layout', layout)
        return layout
    
    @property
    def spatial_shape(self) -> torch.Size:
        spatial_shape = self.get_spatial_cache('shape')
        if spatial_shape is None:
            spatial_shape = self.__cal_spatial_shape(self.coords)
            self.register_spatial_cache('shape', spatial_shape)
        return spatial_shape

    @property
    def feats(self) -> torch.Tensor:
        if config.CONV == 'torchsparse':
            return self.data.F
        elif config.CONV == 'spconv':
            return self.data.features
        else:
            return self.data['feats']
    
    @feats.setter
    def feats(self, value: torch.Tensor):
        if config.CONV == 'torchsparse':
            self.data.F = value
        elif config.CONV == 'spconv':
            self.data.features = value
        else:
            self.data['feats'] = value

    @property
    def coords(self) -> torch.Tensor:
        if config.CONV == 'torchsparse':
            return self.data.C
        elif config.CONV == 'spconv':
            return self.data.indices
        else:
            return self.data['coords']
        
    @coords.setter
    def coords(self, value: torch.Tensor):
        if config.CONV == 'torchsparse':
            self.data.C = value
        elif config.CONV == 'spconv':
            self.data.indices = value
        else:
            self.data['coords'] = value
        if hasattr(self, '_lookup_cache'):
            self._lookup_cache = {}

    @property
    def dtype(self):
        return self.feats.dtype

    @property
    def device(self):
        return self.feats.device
    
    @property
    def seqlen(self) -> torch.LongTensor:
        seqlen = self.get_spatial_cache('seqlen')
        if seqlen is None:
            seqlen = torch.tensor([l.stop - l.start for l in self.layout], dtype=torch.long, device=self.device)
            self.register_spatial_cache('seqlen', seqlen)
        return seqlen
    
    @property
    def cum_seqlen(self) -> torch.LongTensor:
        cum_seqlen = self.get_spatial_cache('cum_seqlen')
        if cum_seqlen is None:
            cum_seqlen = torch.cat([
                torch.tensor([0], dtype=torch.long, device=self.device),
                self.seqlen.cumsum(dim=0)
            ], dim=0)
            self.register_spatial_cache('cum_seqlen', cum_seqlen)
        return cum_seqlen
    
    @property
    def batch_boardcast_map(self) -> torch.LongTensor:
        """
        Get the broadcast map for the varlen tensor.
        """
        batch_boardcast_map = self.get_spatial_cache('batch_boardcast_map')
        if batch_boardcast_map is None:
            batch_boardcast_map = torch.repeat_interleave(
                torch.arange(len(self.layout), device=self.device),
                self.seqlen,
            )
            self.register_spatial_cache('batch_boardcast_map', batch_boardcast_map)
        return batch_boardcast_map

    @overload
    def to(self, dtype: torch.dtype, *, non_blocking: bool = False, copy: bool = False) -> 'SparseTensor': ...

    @overload
    def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None, *, non_blocking: bool = False, copy: bool = False) -> 'SparseTensor': ...

    def to(self, *args, **kwargs) -> 'SparseTensor':
        device = None
        dtype = None
        if len(args) == 2:
            device, dtype = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
        if 'dtype' in kwargs:
            assert dtype is None, "to() received multiple values for argument 'dtype'"
            dtype = kwargs['dtype']
        if 'device' in kwargs:
            assert device is None, "to() received multiple values for argument 'device'"
            device = kwargs['device']
        non_blocking = kwargs.get('non_blocking', False)
        copy = kwargs.get('copy', False)
        
        new_feats = self.feats.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        new_coords = self.coords.to(device=device, non_blocking=non_blocking, copy=copy)
        return self.replace(new_feats, new_coords)

    def type(self, dtype):
        new_feats = self.feats.type(dtype)
        return self.replace(new_feats)

    def cpu(self) -> 'SparseTensor':
        new_feats = self.feats.cpu()
        new_coords = self.coords.cpu()
        return self.replace(new_feats, new_coords)
    
    def cuda(self) -> 'SparseTensor':
        new_feats = self.feats.cuda()
        new_coords = self.coords.cuda()
        return self.replace(new_feats, new_coords)

    def half(self) -> 'SparseTensor':
        new_feats = self.feats.half()
        return self.replace(new_feats)
    
    def float(self) -> 'SparseTensor':
        new_feats = self.feats.float()
        return self.replace(new_feats)
    
    def detach(self) -> 'SparseTensor':
        new_coords = self.coords.detach()
        new_feats = self.feats.detach()
        return self.replace(new_feats, new_coords)

    def reshape(self, *shape) -> 'SparseTensor':
        new_feats = self.feats.reshape(self.feats.shape[0], *shape)
        return self.replace(new_feats)
    
    def unbind(self, dim: int) -> List['SparseTensor']:
        return sparse_unbind(self, dim)

    def replace(self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None) -> 'SparseTensor':
        if config.CONV == 'torchsparse':
            new_data = self.SparseTensorData(
                feats=feats,
                coords=self.data.coords if coords is None else coords,
                stride=self.data.stride,
                spatial_range=self.data.spatial_range,
            )
            new_data._caches = self.data._caches
        elif config.CONV == 'spconv':
            new_data = self.SparseTensorData(
                self.data.features.reshape(self.data.features.shape[0], -1),
                self.data.indices,
                self.data.spatial_shape,
                self.data.batch_size,
                self.data.grid,
                self.data.voxel_num,
                self.data.indice_dict
            )
            new_data._features = feats
            new_data.benchmark = self.data.benchmark
            new_data.benchmark_record = self.data.benchmark_record
            new_data.thrust_allocator = self.data.thrust_allocator
            new_data._timer = self.data._timer
            new_data.force_algo = self.data.force_algo
            new_data.int8_scale = self.data.int8_scale
            if coords is not None:
                new_data.indices = coords
        else:
            new_data = {
                'feats': feats,
                'coords': self.data['coords'] if coords is None else coords,
            }
        new_tensor = SparseTensor(
            new_data,
            shape=torch.Size([self._shape[0]] + list(feats.shape[1:])) if self._shape is not None else None,
            scale=self._scale,
            spatial_cache=self._spatial_cache
        )
        new_tensor._lookup_cache = self._lookup_cache if coords is None else {}
        return new_tensor
    
    def to_dense(self) -> torch.Tensor:
        if config.CONV == 'torchsparse':
            return self.data.dense()
        elif config.CONV == 'spconv':
            return self.data.dense()
        else:
            spatial_shape = self.spatial_shape
            ret = torch.zeros(*self.shape, *spatial_shape, dtype=self.dtype, device=self.device)
            idx = [self.coords[:, 0], slice(None)] + self.coords[:, 1:].unbind(1)
            ret[tuple(idx)] = self.feats
            return ret

    @staticmethod
    def full(aabb, dim, value, dtype=torch.float32, device=None) -> 'SparseTensor':
        N, C = dim
        x = torch.arange(aabb[0], aabb[3] + 1)
        y = torch.arange(aabb[1], aabb[4] + 1)
        z = torch.arange(aabb[2], aabb[5] + 1)
        coords = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).reshape(-1, 3)
        coords = torch.cat([
            torch.arange(N).view(-1, 1).repeat(1, coords.shape[0]).view(-1, 1),
            coords.repeat(N, 1),
        ], dim=1).to(dtype=torch.int32, device=device)
        feats = torch.full((coords.shape[0], C), value, dtype=dtype, device=device)
        return SparseTensor(feats=feats, coords=coords)

    def __merge_sparse_cache(self, other: 'SparseTensor') -> dict:
        new_cache = {}
        for k in set(list(self._spatial_cache.keys()) + list(other._spatial_cache.keys())):
            if k in self._spatial_cache:
                new_cache[k] = self._spatial_cache[k]
            if k in other._spatial_cache:
                if k not in new_cache:
                    new_cache[k] = other._spatial_cache[k]
                else:
                    new_cache[k].update(other._spatial_cache[k])
        return new_cache
    
    def __elemwise__(self, other: Union[torch.Tensor, VarLenTensor], op: callable) -> 'SparseTensor':
        if isinstance(other, torch.Tensor):
            try:
                other = torch.broadcast_to(other, self.shape)
                other = other[self.batch_boardcast_map]
            except:
                pass
        if isinstance(other, VarLenTensor):
            other = other.feats
        new_feats = op(self.feats, other)
        new_tensor = self.replace(new_feats)
        if isinstance(other, SparseTensor):
            new_tensor._spatial_cache = self.__merge_sparse_cache(other)
        return new_tensor

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = range(*idx.indices(self.shape[0]))
        elif isinstance(idx, list):
            assert all(isinstance(i, int) for i in idx), f"Only integer indices are supported: {idx}"
        elif isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                assert idx.shape == (self.shape[0],), f"Invalid index shape: {idx.shape}"
                idx = idx.nonzero().squeeze(1)
            elif idx.dtype in [torch.int32, torch.int64]:
                assert len(idx.shape) == 1, f"Invalid index shape: {idx.shape}"
            else:
                raise ValueError(f"Unknown index type: {idx.dtype}")
        else:
            raise ValueError(f"Unknown index type: {type(idx)}")
        
        new_coords = []
        new_feats = []
        new_layout = []
        new_shape = torch.Size([len(idx)] + list(self.shape[1:]))
        start = 0
        for new_idx, old_idx in enumerate(idx):
            new_coords.append(self.coords[self.layout[old_idx]].clone())
            new_coords[-1][:, 0] = new_idx
            new_feats.append(self.feats[self.layout[old_idx]])
            new_layout.append(slice(start, start + len(new_coords[-1])))
            start += len(new_coords[-1])
        new_coords = torch.cat(new_coords, dim=0).contiguous()
        new_feats = torch.cat(new_feats, dim=0).contiguous()
        new_tensor = SparseTensor(feats=new_feats, coords=new_coords, shape=new_shape)
        new_tensor.register_spatial_cache('layout', new_layout)
        return new_tensor
    
    def clear_spatial_cache(self) -> None:
        """
        Clear all spatial caches.
        """
        self._spatial_cache = {}

    def register_spatial_cache(self, key, value) -> None:
        """
        Register a spatial cache.
        The spatial cache can be any thing you want to cache.
        The registery and retrieval of the cache is based on current scale.
        """
        scale_key = str(self._scale)
        if scale_key not in self._spatial_cache:
            self._spatial_cache[scale_key] = {}
        self._spatial_cache[scale_key][key] = value

    def get_spatial_cache(self, key=None):
        """
        Get a spatial cache.
        """
        scale_key = str(self._scale)
        cur_scale_cache = self._spatial_cache.get(scale_key, {})
        if key is None:
            return cur_scale_cache
        return cur_scale_cache.get(key, None)
    
    def __repr__(self) -> str:
        return f"SparseTensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"

def sparse_cat(inputs: List[SparseTensor], dim: int = 0) -> SparseTensor:
    """
    Concatenate a list of sparse tensors.
    
    Args:
        inputs (List[SparseTensor]): List of sparse tensors to concatenate.
    """
    if dim == 0:
        start = 0
        coords = []
        for input in inputs:
            coords.append(input.coords.clone())
            coords[-1][:, 0] += start
            start += input.shape[0]
        coords = torch.cat(coords, dim=0)
        feats = torch.cat([input.feats for input in inputs], dim=0)
        output = SparseTensor(
            coords=coords,
            feats=feats,
        )
    else:
        feats = torch.cat([input.feats for input in inputs], dim=dim)
        output = inputs[0].replace(feats)

    return output


def _batchwise_cat_rows(inputs: List[torch.Tensor], layouts: List[List[slice]]) -> Tuple[torch.Tensor, List[slice]]:
    batch_size = len(layouts[0])
    chunks = []
    merged_layout = []
    start = 0
    for batch_idx in range(batch_size):
        batch_start = start
        for tensor, layout in zip(inputs, layouts):
            cur = layout[batch_idx]
            if cur.stop > cur.start:
                chunks.append(tensor[cur])
            start += cur.stop - cur.start
        merged_layout.append(slice(batch_start, start))
    if chunks:
        merged = torch.cat(chunks, dim=0).contiguous()
    else:
        merged = inputs[0].new_empty((0, *inputs[0].shape[1:]))
    return merged, merged_layout


def _layout_from_batch_coords(coords: torch.Tensor, batch_size: int) -> List[slice]:
    if coords.shape[0] == 0:
        return [slice(0, 0) for _ in range(batch_size)]
    seq_len = torch.bincount(coords[:, 0], minlength=batch_size)
    offset = torch.cumsum(seq_len, dim=0)
    return [slice((offset[i] - seq_len[i]).item(), offset[i].item()) for i in range(batch_size)]


def _validate_sparse_coords(coords: torch.Tensor, batch_size: int) -> List[slice]:
    if coords.ndim != 2:
        raise ValueError(f"Expected coords to be 2D, got shape {tuple(coords.shape)}")
    if coords.shape[1] < 2:
        raise ValueError(f"Expected coords to have at least 2 columns, got shape {tuple(coords.shape)}")
    if coords.dtype not in [torch.int32, torch.int64]:
        raise ValueError(f"Expected integer coords, got {coords.dtype}")
    if coords.shape[0] == 0:
        return [slice(0, 0) for _ in range(batch_size)]
    batch_idx = coords[:, 0]
    if torch.any(batch_idx < 0) or torch.any(batch_idx >= batch_size):
        raise ValueError(f"Invalid batch index in coords: expected values in [0, {batch_size - 1}]")

    layout = _layout_from_batch_coords(coords, batch_size)
    for i, s in enumerate(layout):
        if s.stop > s.start and not torch.all(coords[s, 0] == i):
            raise ValueError("coords must keep samples from the same batch contiguous")
    return layout


def _coords_to_hash(coords: torch.Tensor, spatial_shape: Sequence[int], batch_size: int) -> torch.Tensor:
    dims = [batch_size, *spatial_shape]
    if len(dims) != coords.shape[1]:
        raise ValueError(f"Coordinate dimension mismatch: got {coords.shape[1]}, expected {len(dims)}")
    offsets = [1] * len(dims)
    for i in range(len(dims) - 2, -1, -1):
        offsets[i] = offsets[i + 1] * int(dims[i + 1])
    offsets = torch.tensor(offsets, dtype=torch.int64, device=coords.device)
    return (coords.to(torch.int64) * offsets.unsqueeze(0)).sum(dim=1)


def _get_sparse_lookup_cache(x: SparseTensor) -> Tuple[torch.Tensor, torch.Tensor]:
    spatial_shape = tuple(int(v) for v in x.spatial_shape)
    cache_key = ('coords_lookup', x.coords.data_ptr(), spatial_shape, int(x.shape[0]))
    cached = x._lookup_cache.get(cache_key, None)
    if cached is None:
        source_hash = _coords_to_hash(x.coords, spatial_shape, x.shape[0])
        order = torch.argsort(source_hash)
        sorted_hash = source_hash[order]
        cached = (sorted_hash, order)
        x._lookup_cache[cache_key] = cached
    return cached


def _lookup_sparse_rows(x: SparseTensor, coords: torch.Tensor) -> Tuple[torch.Tensor, List[slice]]:
    batch_size = x.shape[0]
    layout = _validate_sparse_coords(coords, batch_size)
    if coords.shape[1] != x.coords.shape[1]:
        raise ValueError(f"Coordinate dimension mismatch: got {coords.shape[1]}, expected {x.coords.shape[1]}")

    query_hash = _coords_to_hash(coords, x.spatial_shape, batch_size)

    if query_hash.numel() != torch.unique(query_hash).numel():
        raise ValueError("Query coords contain duplicates, which would create duplicate sparse entries")
    sorted_hash, order = _get_sparse_lookup_cache(x)
    if sorted_hash.numel() > 1 and config.DEBUG:
        if torch.any(sorted_hash[1:] == sorted_hash[:-1]):
            raise ValueError("Input SparseTensor contains duplicate coords, which is not supported")
    if sorted_hash.numel() == 0:
        if query_hash.numel() == 0:
            return query_hash.new_empty((0,), dtype=torch.long), layout
        raise ValueError("Cannot query non-empty coords from an empty SparseTensor")

    pos = torch.searchsorted(sorted_hash, query_hash)
    missing = (pos >= sorted_hash.shape[0]) | (sorted_hash[pos.clamp_max(max(sorted_hash.shape[0] - 1, 0))] != query_hash)
    if torch.any(missing):
        missing_coords = coords[missing]
        raise ValueError(f"Some query coords are not present in the SparseTensor: {missing_coords[:8].tolist()}")

    row_indices = order[pos]
    return row_indices, layout


def _select_channel2spatial_cache(
    cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    row_indices: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    fine_coords, idx, subidx = cache
    if idx.numel() == 0:
        empty_rows = idx.new_empty((0,), dtype=torch.long)
        return (fine_coords, idx, subidx), empty_rows

    if row_indices.numel() == 0:
        kept_fine_rows = idx.new_empty((0,), dtype=torch.long)
        new_fine_coords = fine_coords.new_empty((0, fine_coords.shape[1]))
        new_subidx = subidx.new_empty((0,))
        new_idx = idx.new_empty((0,))
        return (new_fine_coords, new_idx, new_subidx), kept_fine_rows

    max_idx = int(idx.max().item())
    row_indices = row_indices.to(dtype=torch.long, device=idx.device)
    if torch.any(row_indices < 0) or torch.any(row_indices > max_idx):
        raise ValueError(f"Invalid channel2spatial_2 cache: row index out of range [0, {max_idx}]")

    remap = torch.full((max_idx + 1,), -1, dtype=torch.long, device=idx.device)
    remap[row_indices] = torch.arange(row_indices.shape[0], dtype=torch.long, device=idx.device)

    idx_long = idx.to(dtype=torch.long)
    keep_mask = remap[idx_long] >= 0
    kept_fine_rows = keep_mask.nonzero().squeeze(1)

    new_fine_coords = fine_coords[keep_mask].contiguous()
    new_subidx = subidx[keep_mask].contiguous()
    new_idx = remap[idx_long[keep_mask]].to(dtype=idx.dtype).contiguous()

    return (new_fine_coords, new_idx, new_subidx), kept_fine_rows


def _select_preserved_sparse_cache(input: SparseTensor, row_indices: torch.Tensor) -> dict:
    selected_cache = {}
    cur_scale = input._scale
    cur_rows = row_indices.to(dtype=torch.long)

    while True:
        scale_key = str(cur_scale)
        scale_cache = input._spatial_cache.get(scale_key, {})
        new_scale_cache = {}

        if 'shape' in scale_cache:
            new_scale_cache['shape'] = scale_cache['shape']
        if 'subdivision' in scale_cache:
            new_scale_cache['subdivision'] = scale_cache['subdivision'][cur_rows]

        next_rows = None
        if 'channel2spatial_2' in scale_cache:
            new_channel2spatial, next_rows = _select_channel2spatial_cache(
                scale_cache['channel2spatial_2'],
                cur_rows,
            )
            new_scale_cache['channel2spatial_2'] = new_channel2spatial

        if len(new_scale_cache) > 0:
            selected_cache[scale_key] = new_scale_cache

        if next_rows is None:
            break

        cur_scale = tuple([s / 2 for s in cur_scale])
        cur_rows = next_rows

    return selected_cache


def _infer_channel2spatial_layout(cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_size: int) -> List[slice]:
    fine_coords, idx, _ = cache
    fine_layout = _layout_from_batch_coords(fine_coords, batch_size)
    layout = []
    prev_stop = 0
    for batch_idx in range(batch_size):
        cur = fine_layout[batch_idx]
        if cur.stop == cur.start:
            layout.append(slice(prev_stop, prev_stop))
            continue
        cur_idx = idx[cur]
        start = int(cur_idx.min().item())
        stop = int(cur_idx.max().item()) + 1
        if start != prev_stop:
            raise ValueError(f"Invalid channel2spatial_2 cache: batch {batch_idx} is not contiguous")
        if config.DEBUG and cur_idx.unique().numel() != stop - start:
            raise ValueError(f"Invalid channel2spatial_2 cache: batch {batch_idx} has missing coarse indices")
        layout.append(slice(start, stop))
        prev_stop = stop
    return layout


def _merge_channel2spatial_cache(
    caches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    batch_size: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], List[List[slice]]]:
    fine_coords_list = [cache[0] for cache in caches]
    idx_list = [cache[1] for cache in caches]
    subidx_list = [cache[2] for cache in caches]
    fine_layouts = [_layout_from_batch_coords(fine_coords, batch_size) for fine_coords in fine_coords_list]
    coarse_layouts = [_infer_channel2spatial_layout(cache, batch_size) for cache in caches]

    merged_fine_coords, _ = _batchwise_cat_rows(fine_coords_list, fine_layouts)
    merged_subidx, _ = _batchwise_cat_rows(subidx_list, fine_layouts)

    merged_idx_chunks = []
    coarse_start = 0
    for batch_idx in range(batch_size):
        batch_offset = coarse_start
        for fine_layout, coarse_layout, idx in zip(fine_layouts, coarse_layouts, idx_list):
            fine_slice = fine_layout[batch_idx]
            coarse_slice = coarse_layout[batch_idx]
            coarse_len = coarse_slice.stop - coarse_slice.start
            if fine_slice.stop > fine_slice.start:
                merged_idx_chunks.append(idx[fine_slice] - coarse_slice.start + batch_offset)
            batch_offset += coarse_len
        coarse_start = batch_offset

    if merged_idx_chunks:
        merged_idx = torch.cat(merged_idx_chunks, dim=0).contiguous()
    else:
        merged_idx = idx_list[0].new_empty((0,))

    return (merged_fine_coords, merged_idx, merged_subidx), coarse_layouts


def _merge_preserved_sparse_cache(inputs: List[SparseTensor]) -> dict:
    batch_size = inputs[0].shape[0]
    merged_cache = {}
    scale_keys = sorted(set().union(*(tensor._spatial_cache.keys() for tensor in inputs)))

    for scale_key in scale_keys:
        scale_caches = [tensor._spatial_cache.get(scale_key, {}) for tensor in inputs]
        merged_scale_cache = {}

        shapes = [cache['shape'] for cache in scale_caches if 'shape' in cache]
        if len(shapes) > 0:
            ref_shape = torch.Size([
                max(int(shape[d]) for shape in shapes)
                for d in range(len(shapes[0]))
            ])
            merged_scale_cache['shape'] = ref_shape

        coarse_layouts = None
        has_channel2spatial = ['channel2spatial_2' in cache for cache in scale_caches]
        if all(has_channel2spatial):
            merged_channel2spatial, coarse_layouts = _merge_channel2spatial_cache(
                [cache['channel2spatial_2'] for cache in scale_caches],
                batch_size,
            )
            merged_scale_cache['channel2spatial_2'] = merged_channel2spatial

        has_subdivision = ['subdivision' in cache for cache in scale_caches]
        if all(has_subdivision):
            if coarse_layouts is not None:
                merged_subdivision, _ = _batchwise_cat_rows(
                    [cache['subdivision'] for cache in scale_caches],
                    coarse_layouts,
                )
            else:
                subdivision_layouts = [tensor.layout for tensor in inputs]
                merged_subdivision, _ = _batchwise_cat_rows(
                    [cache['subdivision'] for cache in scale_caches],
                    subdivision_layouts,
                )
            merged_scale_cache['subdivision'] = merged_subdivision

        if len(merged_scale_cache) > 0:
            merged_cache[scale_key] = merged_scale_cache

    return merged_cache


def _normalize_sparse_row_indices(
    input: SparseTensor,
    row_indices: Union[torch.Tensor, Sequence[int]],
) -> torch.LongTensor:
    if isinstance(row_indices, torch.Tensor):
        if row_indices.dtype == torch.bool:
            assert row_indices.ndim == 1 and row_indices.shape[0] == input.coords.shape[0], \
                f"Invalid mask shape: {tuple(row_indices.shape)}"
            row_indices = row_indices.nonzero().squeeze(1)
        else:
            assert row_indices.ndim == 1, f"Invalid row index shape: {tuple(row_indices.shape)}"
    else:
        row_indices = torch.tensor(row_indices, dtype=torch.long, device=input.device)

    row_indices = row_indices.to(device=input.device, dtype=torch.long)
    if row_indices.numel() > 0:
        if torch.any(row_indices < 0) or torch.any(row_indices >= input.coords.shape[0]):
            raise ValueError(f"Row indices out of range [0, {input.coords.shape[0] - 1}]")
    return row_indices


def sparse_select_by_rows(
    input: SparseTensor,
    row_indices: Union[torch.Tensor, Sequence[int]],
    *,
    coords: Optional[torch.Tensor] = None,
    layout: Optional[List[slice]] = None,
) -> SparseTensor:
    """
    Select sparse entries by row indices or a boolean mask.

    Args:
        input (SparseTensor): Source sparse tensor.
        row_indices: 1D integer row indices or a boolean mask of shape [T].
        coords: Optional explicit output coords. Use this when row order is meant
            to match an external coordinate tensor exactly.
        layout: Optional explicit output layout. If omitted, it will be inferred
            from `coords` if provided, otherwise from the selected coords.
    """
    row_indices = _normalize_sparse_row_indices(input, row_indices)
    new_feats = input.feats.index_select(0, row_indices).contiguous()
    inferred_coords = input.coords.index_select(0, row_indices).contiguous()
    new_coords = inferred_coords if coords is None else coords.contiguous()
    if layout is None:
        layout = _validate_sparse_coords(new_coords, input.shape[0])
    new_cache = _select_preserved_sparse_cache(input, row_indices)

    output = SparseTensor(
        feats=new_feats,
        coords=new_coords,
        shape=torch.Size([input.shape[0], *input.feats.shape[1:]]),
        scale=input._scale,
        spatial_cache=new_cache,
    )
    output.register_spatial_cache('layout', layout)
    return output


def sparse_batchwise_cat(inputs: List[SparseTensor]) -> SparseTensor:
    """
    Concatenate sparse tensors within each batch while keeping batch size unchanged.

    This helper intentionally preserves only the cache entries that remain valid
    after a same-batch merge: `shape`, `subdivision`, and `channel2spatial_2`.
    Other spatial caches are dropped and can be rebuilt lazily.
    """
    if len(inputs) == 0:
        raise ValueError("sparse_batchwise_cat() expects at least one input")
    if len(inputs) == 1:
        return inputs[0]

    ref = inputs[0]
    batch_size = ref.shape[0]
    feat_shape = ref.feats.shape[1:]
    coord_dim = ref.coords.shape[1]
    ref_scale = ref._scale

    for i, tensor in enumerate(inputs[1:], start=1):
        if tensor.shape[0] != batch_size:
            raise ValueError(f"Input {i} has batch size {tensor.shape[0]}, expected {batch_size}")
        if tensor.feats.shape[1:] != feat_shape:
            raise ValueError(f"Input {i} has feature shape {tensor.feats.shape[1:]}, expected {feat_shape}")
        if tensor.coords.shape[1] != coord_dim:
            raise ValueError(f"Input {i} has coordinate dimension {tensor.coords.shape[1]}, expected {coord_dim}")
        if tensor._scale != ref_scale:
            raise ValueError(f"Input {i} has scale {tensor._scale}, expected {ref_scale}")
        if tensor.device != ref.device:
            raise ValueError(f"Input {i} is on {tensor.device}, expected {ref.device}")

    merged_feats, merged_layout = _batchwise_cat_rows(
        [tensor.feats for tensor in inputs],
        [tensor.layout for tensor in inputs],
    )
    merged_coords, _ = _batchwise_cat_rows(
        [tensor.coords for tensor in inputs],
        [tensor.layout for tensor in inputs],
    )

    merged_cache = _merge_preserved_sparse_cache(inputs)
    output = SparseTensor(
        feats=merged_feats,
        coords=merged_coords,
        shape=torch.Size([batch_size, *feat_shape]),
        scale=ref_scale,
        spatial_cache=merged_cache,
    )
    output.register_spatial_cache('layout', merged_layout)
    return output


def sparse_select_by_coords(input: SparseTensor, coords: torch.Tensor) -> SparseTensor:
    """
    Select sparse entries by their coordinates.

    Args:
        input (SparseTensor): Source sparse tensor.
        coords (torch.Tensor): [N, D+1] integer coordinates to extract. The first
            column is batch index and rows must keep the standard SparseTensor
            batch-contiguous ordering.
    """
    row_indices, layout = _lookup_sparse_rows(input, coords)
    return sparse_select_by_rows(input, row_indices, coords=coords, layout=layout)


def sparse_unbind(input: SparseTensor, dim: int) -> List[SparseTensor]:
    """
    Unbind a sparse tensor along a dimension.
    
    Args:
        input (SparseTensor): Sparse tensor to unbind.
        dim (int): Dimension to unbind.
    """
    if dim == 0:
        return [input[i] for i in range(input.shape[0])]
    else:
        feats = input.feats.unbind(dim)
        return [input.replace(f) for f in feats]
