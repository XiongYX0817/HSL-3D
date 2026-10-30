from typing import *

import torch
import torch.nn as nn


class RotaryPositionEmbedder(nn.Module):
    """
    Drop-in real-valued replacement for the original complex-number RoPE helper.

    Public API is intentionally kept identical to `rope.py`:
    - same class name
    - same constructor signature
    - same `forward(indices)` signature
    - same `apply_rotary_embedding(x, phases)` entry point

    The only difference is the internal representation of `phases`:
    this module stores `[cos(theta), sin(theta)]` pairs in the last dimension
    instead of complex tensors.
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

    @staticmethod
    def apply_rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [..., N, H, D] or compatible tensor with even last dim
            phases (torch.Tensor): [..., N, D/2, 2] where the last dim stores
                [cos(theta), sin(theta)].
        """
        assert x.shape[-1] % 2 == 0, "Last dim of x must be even for RoPE"
        assert phases.shape[-1] == 2, "Last dim of phases must store [cos, sin]"

        x_dtype = x.dtype
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]

        cos = phases[..., 0].to(dtype=x_pairs.dtype)
        sin = phases[..., 1].to(dtype=x_pairs.dtype)
        cos, sin = RotaryPositionEmbedder._broadcast_cos_sin(cos, sin, x_even.ndim)

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        x_embed = torch.stack([out_even, out_odd], dim=-1).reshape(*x.shape).to(x_dtype)
        return x_embed

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices (torch.Tensor): [..., N, C] tensor of spatial positions

        Returns:
            torch.Tensor: [..., N, head_dim // 2, 2] real-valued cosine/sine pairs
        """
        assert indices.shape[-1] == self.dim, f"Last dim of indices must be {self.dim}"
        angles = self._get_angles(indices.reshape(-1))
        phases = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        phases = phases.reshape(*indices.shape[:-1], -1, 2)
        phases = self._pad_identity_pairs(phases, self.head_dim // 2)
        return phases
