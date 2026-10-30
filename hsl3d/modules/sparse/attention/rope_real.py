from typing import *

import torch
import torch.nn as nn

from ..basic import SparseTensor


class SparseRotaryPositionEmbedder(nn.Module):
    """
    Drop-in real-valued replacement for the original sparse complex-number RoPE helper.

    Public API is intentionally kept identical to `rope.py`:
    - same class name
    - same constructor signature
    - same `forward(q, k=None)` signature

    The cached `phases` tensor now stores `[cos(theta), sin(theta)]` pairs in the
    last dimension instead of complex values.
    """

    def __init__(
        self,
        head_dim: int,
        dim: int = 3,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
    ):
        super().__init__()
        assert head_dim % 2 == 0, "Head dim must be divisible by 2"
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = rope_freq[0] / (rope_freq[1] ** self.freqs)

    def _get_angles(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        return torch.outer(indices, self.freqs)

    @staticmethod
    def _pad_identity_pairs(phases: torch.Tensor, target_pairs: int) -> torch.Tensor:
        if phases.shape[-2] >= target_pairs:
            return phases
        padn = target_pairs - phases.shape[-2]
        pad_shape = (*phases.shape[:-2], padn, 2)
        pad = torch.zeros(pad_shape, dtype=phases.dtype, device=phases.device)
        pad[..., 0] = 1.0
        return torch.cat([phases, pad], dim=-2)

    @staticmethod
    def _broadcast_cos_sin(
        cos: torch.Tensor,
        sin: torch.Tensor,
        target_ndim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        while cos.ndim < target_ndim:
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
        return cos, sin

    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % 2 == 0, "Last dim of x must be even for RoPE"
        assert phases.shape[-1] == 2, "Last dim of phases must store [cos, sin]"

        x_dtype = x.dtype
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]

        cos = phases[..., 0].to(dtype=x_pairs.dtype)
        sin = phases[..., 1].to(dtype=x_pairs.dtype)
        cos, sin = self._broadcast_cos_sin(cos, sin, x_even.ndim)

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack([out_even, out_odd], dim=-1).reshape(*x.shape).to(x_dtype)

    def forward(self, q: SparseTensor, k: Optional[SparseTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (SparseTensor): [..., N, H, D] tensor of queries
            k (SparseTensor): [..., N, H, D] tensor of keys
        """
        assert q.coords.shape[-1] == self.dim + 1, "Last dimension of coords must be equal to dim+1"
        phases_cache_name = f"rope_phase_{self.dim}d_freq{self.rope_freq[0]}-{self.rope_freq[1]}_hd{self.head_dim}"
        phases = q.get_spatial_cache(phases_cache_name)
        if phases is None:
            coords = q.coords[..., 1:]
            angles = self._get_angles(coords.reshape(-1))
            phases = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            phases = phases.reshape(*coords.shape[:-1], -1, 2)
            phases = self._pad_identity_pairs(phases, self.head_dim // 2)
            q.register_spatial_cache(phases_cache_name, phases)
        q_embed = q.replace(self._rotary_embedding(q.feats, phases))
        if k is None:
            return q_embed
        k_embed = k.replace(self._rotary_embedding(k.feats, phases))
        return q_embed, k_embed
