from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules import sparse as sp
from ...modules.norm import LayerNorm32
from ...modules.sparse.transformer.blocks import SparseFeedForwardNet
from ...modules.utils import convert_module_to_f16, convert_module_to_f32
from .sparse_blocks import (
    SparseConvNeXtBlock3d,
    SparseResBlockC2S3d,
    SparseResBlockS2C3d,
)


_SPARSE_BLOCKS = {
    cls.__name__: cls
    for cls in (
        SparseConvNeXtBlock3d,
        SparseResBlockC2S3d,
        SparseResBlockS2C3d,
    )
}


class SparseCrossAttentionBlock(nn.Module):
    """Cross-attention followed by an MLP, without query self-attention."""

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.cross_attn = sp.SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)

    def _forward(self, x: sp.SparseTensor, context: sp.SparseTensor) -> sp.SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.mlp(h)
        return x + h

    def forward(self, x: sp.SparseTensor, context: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                x,
                context,
                use_reentrant=False,
            )
        return self._forward(x, context)


class SparseOvoxelFeatureEncoder(nn.Module):
    """Paper Stage II SparseConv stem: 1024-grid O-Voxel to 64-grid features."""

    def __init__(
        self,
        in_channels: int = 6,
        model_channels: Sequence[int] = (64, 128, 256, 512, 1024),
        num_blocks: Sequence[int] = (4, 8, 16, 4),
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
    ):
        super().__init__()
        if len(model_channels) != 5:
            raise ValueError("model_channels must contain the five paper channel widths")
        if len(num_blocks) != 4:
            raise ValueError("num_blocks must contain four ConvNeXt stage depths")

        self.in_channels = in_channels
        self.model_channels = tuple(int(v) for v in model_channels)
        self.num_blocks = tuple(int(v) for v in num_blocks)
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = sp.SparseLinear(in_channels, self.model_channels[0])
        self.stages = nn.ModuleList()
        for stage_idx, depth in enumerate(self.num_blocks):
            stage = nn.ModuleList([
                SparseResBlockS2C3d(
                    self.model_channels[stage_idx],
                    self.model_channels[stage_idx + 1],
                    use_checkpoint=use_checkpoint,
                )
            ])
            stage.extend([
                SparseConvNeXtBlock3d(
                    self.model_channels[stage_idx + 1],
                    mlp_ratio=mlp_ratio,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(depth)
            ])
            self.stages.append(stage)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def out_channels(self) -> int:
        return self.model_channels[-1]

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def convert_to_fp16(self) -> None:
        self.dtype = torch.float16
        self.stages.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.dtype = torch.float32
        self.stages.apply(convert_module_to_f32)

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.input_layer(x).type(self.dtype)
        for stage in self.stages:
            for block in stage:
                h = block(h)
        return h


class SparseQueryDecoderTail(nn.Module):
    """Sparse convolutional tail applied after active query selection."""

    def __init__(
        self,
        out_channels: int,
        model_channels: List[int],
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, object]],
        pred_subdiv: bool = True,
    ):
        super().__init__()
        if not (
            len(model_channels)
            == len(num_blocks)
            == len(block_type)
            == len(block_args)
        ):
            raise ValueError("Tail stage configuration lengths must match")
        if len(up_block_type) != len(model_channels) - 1:
            raise ValueError("up_block_type must have one entry per upsample transition")

        self.out_channels = out_channels
        self.model_channels = list(model_channels)
        self.num_blocks = list(num_blocks)
        self.pred_subdiv = pred_subdiv
        self.output_layer = sp.SparseLinear(model_channels[-1], out_channels)

        self.blocks = nn.ModuleList()
        for stage_idx, depth in enumerate(num_blocks):
            if block_type[stage_idx] not in _SPARSE_BLOCKS:
                raise ValueError(f"Unsupported sparse block: {block_type[stage_idx]}")
            stage = nn.ModuleList([
                _SPARSE_BLOCKS[block_type[stage_idx]](
                    model_channels[stage_idx],
                    **block_args[stage_idx],
                )
                for _ in range(depth)
            ])
            if stage_idx < len(num_blocks) - 1:
                if up_block_type[stage_idx] not in _SPARSE_BLOCKS:
                    raise ValueError(f"Unsupported sparse upsample block: {up_block_type[stage_idx]}")
                stage.append(
                    _SPARSE_BLOCKS[up_block_type[stage_idx]](
                        model_channels[stage_idx],
                        model_channels[stage_idx + 1],
                        pred_subdiv=pred_subdiv,
                        **block_args[stage_idx],
                    )
                )
            self.blocks.append(stage)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    @staticmethod
    def _module_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
        for param in module.parameters():
            if torch.is_floating_point(param):
                return param.dtype
        for buffer in module.buffers():
            if torch.is_floating_point(buffer):
                return buffer.dtype
        return fallback

    def forward(self, x: sp.SparseTensor, return_subs: bool = False):
        h = x
        subs_gt = []
        subs = []
        for stage_idx, stage in enumerate(self.blocks):
            for block_idx, block in enumerate(stage):
                block_dtype = self._module_dtype(block, h.dtype)
                if h.dtype != block_dtype:
                    h = h.type(block_dtype)
                is_upsample = stage_idx < len(self.blocks) - 1 and block_idx == len(stage) - 1
                if is_upsample:
                    if self.pred_subdiv:
                        if self.training:
                            subdiv_gt = h.get_spatial_cache("subdivision")
                            if subdiv_gt is None:
                                raise ValueError(
                                    "Training decoder tail requires subdivision cache on the query tensor"
                                )
                            subs_gt.append(subdiv_gt)
                        h, sub = block(h)
                        subs.append(sub)
                    else:
                        h = block(h, subdiv=None)
                else:
                    h = block(h)

        out_dtype = self._module_dtype(self.output_layer, h.dtype)
        if h.dtype != out_dtype:
            h = h.type(out_dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.output_layer(h)
        if self.training and self.pred_subdiv:
            return h, subs_gt, subs
        if return_subs:
            return h, subs
        return h
