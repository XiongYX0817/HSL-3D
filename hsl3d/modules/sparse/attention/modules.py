from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import VarLenTensor, SparseTensor
from .. import config as sparse_config
from .full_attn import sparse_scaled_dot_product_attention
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention
from .rope import SparseRotaryPositionEmbedder


class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: Union[VarLenTensor, torch.Tensor]) -> Union[VarLenTensor, torch.Tensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, VarLenTensor):
            x = x.replace(F.normalize(x.feats, dim=-1) * self.gamma * self.scale)
        else:
            x = F.normalize(x, dim=-1) * self.gamma * self.scale
        return x.to(x_type)


class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed", "double_windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed", "double_windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        assert type == "self" or use_rope is False, "Rotary position embeddings only supported for self-attention"
        if attn_mode == 'double_windowed':
            assert window_size % 2 == 0, "Window size must be even for double windowed attention"
            assert num_heads % 2 == 0, "Number of heads must be even for double windowed attention"
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        
        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = SparseRotaryPositionEmbedder(self.head_dim, rope_freq=rope_freq)

    @staticmethod
    def _linear(module: nn.Linear, x: Union[VarLenTensor, torch.Tensor]) -> Union[VarLenTensor, torch.Tensor]:
        weight_dtype = module.weight.dtype
        if isinstance(x, VarLenTensor):
            feats = x.feats
            if feats.dtype != weight_dtype and not torch.is_autocast_enabled():
                feats = feats.to(dtype=weight_dtype)
            return x.replace(module(feats))
        else:
            if x.dtype != weight_dtype and not torch.is_autocast_enabled():
                x = x.to(dtype=weight_dtype)
            return module(x)

    @staticmethod
    def _reshape_chs(x: Union[VarLenTensor, torch.Tensor], shape: Tuple[int, ...]) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, VarLenTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(self, x: Union[VarLenTensor, torch.Tensor], num_fused: int) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, VarLenTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, VarLenTensor) else x_feats

    @staticmethod
    def _get_attention_target_dtype(device: torch.device, current_dtype: torch.dtype) -> torch.dtype:
        if current_dtype in (torch.float16, torch.bfloat16):
            return current_dtype

        if torch.is_autocast_enabled():
            get_autocast_dtype = getattr(torch, "get_autocast_dtype", None)
            if callable(get_autocast_dtype):
                try:
                    target_dtype = get_autocast_dtype(device.type)
                    if target_dtype in (torch.float16, torch.bfloat16):
                        return target_dtype
                except TypeError:
                    pass
            if device.type == "cuda":
                get_autocast_gpu_dtype = getattr(torch, "get_autocast_gpu_dtype", None)
                if callable(get_autocast_gpu_dtype):
                    target_dtype = get_autocast_gpu_dtype()
                    if target_dtype in (torch.float16, torch.bfloat16):
                        return target_dtype

        if device.type == "cuda":
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(is_bf16_supported) and is_bf16_supported():
                return torch.bfloat16
            return torch.float16

        # Keep the non-CUDA fallback simple. This still avoids float32 inputs
        # for attention backends that only accept low-precision dtypes.
        return torch.bfloat16

    @classmethod
    def _cast_attention_input(cls, x: Union[VarLenTensor, torch.Tensor]) -> Union[VarLenTensor, torch.Tensor]:
        if sparse_config.ATTN not in ("flash_attn", "flash_attn_3"):
            return x

        if isinstance(x, VarLenTensor):
            target_dtype = cls._get_attention_target_dtype(x.device, x.dtype)
            if x.dtype == target_dtype:
                return x
            return x.replace(x.feats.to(dtype=target_dtype))

        target_dtype = cls._get_attention_target_dtype(x.device, x.dtype)
        if x.dtype == target_dtype:
            return x
        return x.to(dtype=target_dtype)
    
    def forward(self, x: SparseTensor, context: Optional[Union[VarLenTensor, torch.Tensor]] = None) -> SparseTensor:
        if self._type == "self":
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            if self.qk_rms_norm or self.use_rope:
                q, k, v = qkv.unbind(dim=-3)
                if self.qk_rms_norm:
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                if self.use_rope:
                    q, k = self.rope(q, k)
                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
            if self.attn_mode == "full":
                qkv = self._cast_attention_input(qkv)
                h = sparse_scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                qkv = self._cast_attention_input(qkv)
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv, self.window_size, shift_window=self.shift_window
                )
            elif self.attn_mode == "double_windowed":
                qkv0 = qkv.replace(qkv.feats[:, :, self.num_heads//2:])
                qkv1 = qkv.replace(qkv.feats[:, :, :self.num_heads//2])
                qkv0 = self._cast_attention_input(qkv0)
                qkv1 = self._cast_attention_input(qkv1)
                h0 = sparse_windowed_scaled_dot_product_self_attention(
                    qkv0, self.window_size, shift_window=(0, 0, 0)
                )
                h1 = sparse_windowed_scaled_dot_product_self_attention(
                    qkv1, self.window_size, shift_window=tuple([self.window_size//2] * 3)
                )
                h = qkv.replace(torch.cat([h0.feats, h1.feats], dim=1))
        else:
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=-3)
                k = self.k_rms_norm(k)
                q = self._cast_attention_input(q)
                k = self._cast_attention_input(k)
                v = self._cast_attention_input(v)
                h = sparse_scaled_dot_product_attention(q, k, v)
            else:
                q = self._cast_attention_input(q)
                kv = self._cast_attention_input(kv)
                h = sparse_scaled_dot_product_attention(q, kv)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        return h
