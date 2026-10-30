import math
from typing import *

import torch
import torch.nn as nn
from ..attention import MultiHeadAttention
from ..norm import LayerNorm32


class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """
    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed


class FourierPositionEmbedder(nn.Module):
    r"""Standard Fourier positional embedding used by the Stage II VAE.

    The frequency bands follow :math:`\omega_k = 2^k\pi`. For an input
    position ``p``, the output concatenates ``sin(omega_k * p)`` and
    ``cos(omega_k * p)`` over all frequency bands.
    """

    def __init__(self, num_frequencies: int, in_channels: int = 3):
        super().__init__()
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

        self.num_frequencies = num_frequencies
        self.in_channels = in_channels
        self.out_channels = 2 * num_frequencies * in_channels
        frequencies = torch.pi * torch.pow(
            2.0,
            torch.arange(num_frequencies, dtype=torch.float32),
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim < 2:
            raise ValueError("positions must have shape (..., in_channels)")
        if positions.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected positions with {self.in_channels} channels, "
                f"got {positions.shape[-1]}"
            )
        if not positions.is_floating_point():
            raise TypeError("positions must be a floating-point tensor")

        input_dtype = positions.dtype
        compute_dtype = (
            torch.float32
            if input_dtype in (torch.float16, torch.bfloat16)
            else input_dtype
        )
        positions = positions.to(dtype=compute_dtype)
        leading_shape = positions.shape[:-1]
        frequency_shape = [1] * len(leading_shape) + [self.num_frequencies, 1]
        frequencies = self.frequencies.to(
            device=positions.device,
            dtype=compute_dtype,
        ).reshape(frequency_shape)
        angles = positions.unsqueeze(-2) * frequencies
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return embedding.flatten(start_dim=-2).to(dtype=input_dtype)


class ScaleModulatedPositionEmbedder(nn.Module):
    r"""
    Fourier positional embedding modulated by the spatial extent of each token.

    For frequency bands :math:`\omega_k = 2^k\pi`, the embedding is

    .. math::
        \operatorname{SMPE}(p, s) = \bigoplus_{k=0}^{K-1}
        [\alpha(\omega_k, s)\sin(\omega_k p),
         \alpha(\omega_k, s)\cos(\omega_k p)],

    where :math:`\alpha(\omega, s) = \exp(-\lambda s^2\omega^2)` and
    :math:`\lambda > 0` is learnable. The output dimension is
    ``2 * num_frequencies * in_channels``.

    Args:
        num_frequencies: Number of frequency bands ``K``.
        in_channels: Spatial dimension of the input positions.
        lambda_init: Initial value of the positive modulation strength.
    """

    def __init__(
        self,
        num_frequencies: int,
        in_channels: int = 3,
        lambda_init: float = 1.0,
    ):
        super().__init__()
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if lambda_init <= 0:
            raise ValueError("lambda_init must be positive")

        self.num_frequencies = num_frequencies
        self.in_channels = in_channels
        self.out_channels = 2 * num_frequencies * in_channels

        frequencies = torch.pi * torch.pow(
            2.0,
            torch.arange(num_frequencies, dtype=torch.float32),
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.log_lambda = nn.Parameter(
            torch.tensor(math.log(lambda_init), dtype=torch.float32)
        )

    @property
    def modulation_strength(self) -> torch.Tensor:
        r"""Return the positive learnable parameter :math:`\lambda`."""
        return self.log_lambda.exp()

    def forward(self, positions: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: Spatial positions with shape ``(..., in_channels)``.
            scales: Positive scalar extents with shape ``(...)`` or ``(..., 1)``.

        Returns:
            Scale-modulated positional embeddings with shape
            ``(..., out_channels)``.
        """
        if positions.ndim < 2:
            raise ValueError("positions must have shape (..., in_channels)")
        if positions.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected positions with {self.in_channels} channels, "
                f"got {positions.shape[-1]}"
            )

        leading_shape = positions.shape[:-1]
        if scales.shape == leading_shape:
            scales = scales.unsqueeze(-1)
        elif scales.shape != (*leading_shape, 1):
            raise ValueError(
                f"Expected scales with shape {leading_shape} or "
                f"{(*leading_shape, 1)}, got {tuple(scales.shape)}"
            )

        if not positions.is_floating_point() or not scales.is_floating_point():
            raise TypeError("positions and scales must be floating-point tensors")

        input_dtype = positions.dtype
        compute_dtype = (
            torch.float32
            if input_dtype in (torch.float16, torch.bfloat16)
            else input_dtype
        )
        positions = positions.to(dtype=compute_dtype)
        scales = scales.to(device=positions.device, dtype=compute_dtype)
        frequencies = self.frequencies.to(device=positions.device, dtype=compute_dtype)
        modulation_strength = self.modulation_strength.to(
            device=positions.device,
            dtype=compute_dtype,
        )

        frequency_shape = [1] * len(leading_shape) + [self.num_frequencies, 1]
        frequencies = frequencies.reshape(frequency_shape)
        angles = positions.unsqueeze(-2) * frequencies
        modulation = torch.exp(
            -modulation_strength * scales.unsqueeze(-2).square() * frequencies.square()
        )

        embedding = torch.cat(
            [modulation * torch.sin(angles), modulation * torch.cos(angles)],
            dim=-1,
        )
        embedding = embedding.flatten(start_dim=-2)
        return embedding.to(dtype=input_dtype)


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN).
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[int] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )

    def _forward(self, x: torch.Tensor, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h, phases=phases)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, phases, use_reentrant=False)
        else:
            return self._forward(x, phases)


class TransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN).
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0), 
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )

    def _forward(self, x: torch.Tensor, context: torch.Tensor, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.self_attn(h, phases=phases)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, context: torch.Tensor, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, context, phases, use_reentrant=False)
        else:
            return self._forward(x, context, phases)
        
