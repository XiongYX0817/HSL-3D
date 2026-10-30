from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import sparse as sp
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules.utils import convert_module_to, manual_cast, str_to_dtype
from ..utils.multiscale_shape_latent_utils import LATENT_LEVELS, latent_level_tensor_to_levels
from .sparse_elastic_mixin import SparseTransformerElasticMixin
from .octree_structure_flow import TimestepEmbedder
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock


_INFERENCE_LEVEL_SLICE_CACHE_KEY = "multiscale_level_slices"


class MultiScaleShapeFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = "float32",
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = "vanilla",
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = str_to_dtype(dtype)
        self.latent_levels = LATENT_LEVELS

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.level_input_projs = nn.ModuleList([
            nn.Linear(in_channels, model_channels)
            for _ in self.latent_levels
        ])
        self.level_embeddings = nn.Embedding(len(self.latent_levels), model_channels)
        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode="full",
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                rope_freq=rope_freq,
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])
        self.level_output_projs = nn.ModuleList([
            nn.Linear(model_channels, out_channels)
            for _ in self.latent_levels
        ])

        self.initialize_weights()
        self.convert_to(self.dtype)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.blocks.apply(partial(convert_module_to, dtype=dtype))

    def initialize_weights(self) -> None:
        if self.initialization == "vanilla":
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

            self.apply(_basic_init)
            nn.init.normal_(self.level_embeddings.weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            for proj in self.level_output_projs:
                nn.init.constant_(proj.weight, 0)
                nn.init.constant_(proj.bias, 0)

        elif self.initialization == "scaled":
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=np.sqrt(2.0 / (5.0 * self.model_channels)))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

            self.apply(_basic_init)
            nn.init.normal_(self.level_embeddings.weight, std=0.02)

            def _scaled_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=1.0 / np.sqrt(5 * self.num_blocks * self.model_channels))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

            for block in self.blocks:
                block.self_attn.to_out.apply(_scaled_init)
                block.cross_attn.to_out.apply(_scaled_init)
                block.mlp.mlp[2].apply(_scaled_init)

            for proj in self.level_input_projs:
                nn.init.normal_(proj.weight, std=1.0 / np.sqrt(self.in_channels))
                nn.init.zeros_(proj.bias)

            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            for proj in self.level_output_projs:
                nn.init.constant_(proj.weight, 0)
                nn.init.constant_(proj.bias, 0)

    def _apply_levelwise_linear(
        self,
        x: sp.SparseTensor,
        latent_levels: torch.Tensor,
        layers: nn.ModuleList,
        out_channels: int,
    ) -> Tuple[sp.SparseTensor, Optional[torch.Tensor]]:
        feats = x.feats
        # Under autocast, layer outputs may not match feats.dtype.
        out = None
        ddp_guard = None
        assigned = torch.zeros((feats.shape[0],), dtype=torch.bool, device=feats.device)
        for level, layer in zip(self.latent_levels, layers):
            mask = latent_levels == level
            if not torch.any(mask):
                # Keep level-specific params in the autograd graph even when a
                # level is absent from a microbatch, so DDP can stay in the fast
                # path with find_unused_parameters=False.
                cur_guard = layer.weight.reshape(-1)[0] * 0
                if layer.bias is not None:
                    cur_guard = cur_guard + layer.bias.reshape(-1)[0] * 0
                ddp_guard = cur_guard if ddp_guard is None else ddp_guard + cur_guard
                continue
            cur_out = layer(feats[mask])
            if out is None:
                out = cur_out.new_zeros((feats.shape[0], out_channels))
            elif cur_out.dtype != out.dtype:
                cur_out = cur_out.to(dtype=out.dtype)
            out[mask] = cur_out
            assigned[mask] = True
        if not torch.all(assigned):
            invalid_levels = torch.unique(latent_levels[~assigned]).tolist()
            raise ValueError(f"Unsupported latent levels encountered: {invalid_levels}")
        if out is None:
            out = feats.new_zeros((feats.shape[0], out_channels))
        return x.replace(out), ddp_guard

    def _build_level_run_slices(self, latent_levels: torch.Tensor) -> Dict[int, Tuple[slice, ...]]:
        latent_levels = latent_levels.to(dtype=torch.long)
        level_slices: Dict[int, List[slice]] = {
            int(level): []
            for level in self.latent_levels
        }
        if latent_levels.numel() == 0:
            return {}

        run_starts = torch.ones((latent_levels.shape[0],), dtype=torch.bool, device=latent_levels.device)
        run_starts[1:] = latent_levels[1:] != latent_levels[:-1]
        starts = torch.nonzero(run_starts, as_tuple=False).squeeze(1)
        stops = torch.cat([starts[1:], latent_levels.new_tensor([latent_levels.shape[0]])], dim=0)
        run_levels = latent_levels.index_select(0, starts)

        for level, start, stop in zip(run_levels.tolist(), starts.tolist(), stops.tolist()):
            if level not in level_slices:
                raise ValueError(f"Unsupported latent level encountered: {level}")
            level_slices[level].append(slice(start, stop))

        return {
            latent_level: tuple(slices)
            for latent_level, slices in level_slices.items()
            if len(slices) > 0
        }

    def _get_inference_level_slices(
        self,
        x: sp.SparseTensor,
        latent_level: Union[sp.VarLenTensor, torch.Tensor],
        latent_levels: torch.Tensor,
    ) -> Optional[Dict[int, Tuple[slice, ...]]]:
        if torch.is_grad_enabled():
            return None

        if isinstance(latent_level, sp.VarLenTensor):
            cached = latent_level._cache.get(_INFERENCE_LEVEL_SLICE_CACHE_KEY)
            if cached is not None:
                return cached

        cached = x.get_spatial_cache(_INFERENCE_LEVEL_SLICE_CACHE_KEY)
        if cached is not None:
            if isinstance(latent_level, sp.VarLenTensor):
                latent_level._cache[_INFERENCE_LEVEL_SLICE_CACHE_KEY] = cached
            return cached

        computed = self._build_level_run_slices(latent_levels)
        if isinstance(latent_level, sp.VarLenTensor):
            latent_level._cache[_INFERENCE_LEVEL_SLICE_CACHE_KEY] = computed
        return computed

    def _apply_levelwise_linear_fast(
        self,
        x: sp.SparseTensor,
        level_slices: Dict[int, Tuple[slice, ...]],
        layers: nn.ModuleList,
        out_channels: int,
    ) -> sp.SparseTensor:
        feats = x.feats
        out = None

        for level, layer in zip(self.latent_levels, layers):
            segments = level_slices.get(int(level), ())
            if len(segments) == 0:
                continue

            if len(segments) == 1:
                cur_out = layer(feats[segments[0]])
                valid_segments = segments
                lengths = (segments[0].stop - segments[0].start,)
            else:
                chunks = []
                valid_segments = []
                lengths = []
                for seg in segments:
                    seg_len = seg.stop - seg.start
                    if seg_len <= 0:
                        continue
                    chunks.append(feats[seg])
                    valid_segments.append(seg)
                    lengths.append(seg_len)
                if len(chunks) == 0:
                    continue
                cur_out = layer(torch.cat(chunks, dim=0))
                lengths = tuple(lengths)
                valid_segments = tuple(valid_segments)

            if out is None:
                out = cur_out.new_zeros((feats.shape[0], out_channels))
            elif cur_out.dtype != out.dtype:
                cur_out = cur_out.to(dtype=out.dtype)

            if len(valid_segments) == 1:
                out[valid_segments[0]] = cur_out
            else:
                offset = 0
                for seg, seg_len in zip(valid_segments, lengths):
                    out[seg] = cur_out[offset:offset + seg_len]
                    offset += seg_len

        if out is None:
            out = feats.new_zeros((feats.shape[0], out_channels))
        return x.replace(out)

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, List[torch.Tensor]],
        latent_level: Union[sp.VarLenTensor, torch.Tensor],
        concat_cond: Optional[sp.SparseTensor] = None,
        **kwargs,
    ) -> sp.SparseTensor:
        del kwargs
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)

        latent_levels = latent_level_tensor_to_levels(latent_level).to(device=x.device)
        if latent_levels.shape[0] != x.coords.shape[0]:
            raise ValueError(f"latent_level length mismatch: got {latent_levels.shape[0]}, expected {x.coords.shape[0]}")

        level_slices = self._get_inference_level_slices(x, latent_level, latent_levels)
        ddp_guard = None
        if level_slices is None:
            h, cur_guard = self._apply_levelwise_linear(x, latent_levels, self.level_input_projs, self.model_channels)
            if cur_guard is not None:
                ddp_guard = cur_guard
        else:
            h = self._apply_levelwise_linear_fast(x, level_slices, self.level_input_projs, self.model_channels)
        level_emb = self.level_embeddings(latent_levels).to(dtype=h.feats.dtype)
        h = h.replace(h.feats + level_emb)
        h = manual_cast(h, self.dtype)

        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        cond = manual_cast(cond, self.dtype)

        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:].float())
            h = h + manual_cast(pe, self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)

        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        if level_slices is None:
            h, cur_guard = self._apply_levelwise_linear(h, latent_levels, self.level_output_projs, self.out_channels)
            if cur_guard is not None:
                ddp_guard = cur_guard if ddp_guard is None else ddp_guard + cur_guard
            if ddp_guard is not None:
                h = h.replace(h.feats + ddp_guard.to(dtype=h.feats.dtype))
        else:
            h = self._apply_levelwise_linear_fast(h, level_slices, self.level_output_projs, self.out_channels)
        return h


class ElasticMultiScaleShapeFlowModel(SparseTransformerElasticMixin, MultiScaleShapeFlowModel):
    """
    Multiscale shape-latent flow model with elastic memory management.
    """

    pass
