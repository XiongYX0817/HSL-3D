import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from o_voxel.convert import flexible_dual_grid_to_mesh

from ...modules import sparse as sp
from ...modules.sparse.transformer.blocks import SparseTransformerBlock
from ...modules.transformer import FourierPositionEmbedder, ScaleModulatedPositionEmbedder
from ...representations import Mesh
from ...utils.multiscale_shape_latent_utils import (
    LATENT_LEVELS,
    build_level_sparse_tensor,
    local_coords_to_storage_coords,
    latent_level_grid_resolution,
    latent_level_spatial_extent,
)
from .multiscale_shape_vae_blocks import (
    SparseCrossAttentionBlock,
    SparseOvoxelFeatureEncoder,
    SparseQueryDecoderTail,
)


def _coords_to_hash(
    coords: torch.Tensor,
    spatial_shape: Sequence[int],
    batch_size: int,
) -> torch.Tensor:
    dims = [batch_size, *spatial_shape]
    if len(dims) != coords.shape[1]:
        raise ValueError(
            f"Coordinate dimension mismatch: got {coords.shape[1]}, expected {len(dims)}"
        )
    offsets = [1] * len(dims)
    for idx in range(len(dims) - 2, -1, -1):
        offsets[idx] = offsets[idx + 1] * int(dims[idx + 1])
    offsets = torch.tensor(offsets, dtype=torch.int64, device=coords.device)
    return (coords.to(torch.int64) * offsets.unsqueeze(0)).sum(dim=1)


def _sparse_coords_membership_mask(
    source_coords: torch.Tensor,
    query_coords: torch.Tensor,
) -> torch.Tensor:
    if source_coords.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.bool, device=source_coords.device)
    if query_coords.shape[0] == 0:
        return torch.zeros(
            (source_coords.shape[0],),
            dtype=torch.bool,
            device=source_coords.device,
        )

    max_spatial = torch.maximum(
        source_coords[:, 1:].amax(dim=0),
        query_coords[:, 1:].amax(dim=0),
    ) + 1
    batch_size = int(
        max(source_coords[:, 0].max().item(), query_coords[:, 0].max().item())
    ) + 1
    query_hash = _coords_to_hash(query_coords, max_spatial.tolist(), batch_size)
    source_hash = _coords_to_hash(source_coords, max_spatial.tolist(), batch_size)
    sorted_query = torch.sort(query_hash).values
    positions = torch.searchsorted(sorted_query, source_hash)
    in_range = positions < sorted_query.shape[0]
    clamped = positions.clamp_max(max(sorted_query.shape[0] - 1, 0))
    return in_range & (sorted_query[clamped] == source_hash)


def _align_sparse_to_reference(
    source: sp.SparseTensor,
    reference: sp.SparseTensor,
) -> sp.SparseTensor:
    if torch.equal(source.coords, reference.coords):
        return source
    if reference.coords.shape[0] == 0:
        empty_feats = source.feats.new_empty((0, *source.feats.shape[1:]))
        return reference.replace(empty_feats + source.feats.sum() * 0)
    if source.coords.shape[0] == 0:
        zero_feats = source.feats.new_zeros(
            (reference.coords.shape[0], *source.feats.shape[1:])
        )
        return reference.replace(zero_feats + source.feats.sum() * 0)

    spatial_shape = (
        torch.maximum(
            source.coords[:, 1:].amax(dim=0),
            reference.coords[:, 1:].amax(dim=0),
        )
        + 1
    ).tolist()
    source_hash = _coords_to_hash(source.coords, spatial_shape, source.shape[0])
    reference_hash = _coords_to_hash(
        reference.coords,
        spatial_shape,
        reference.shape[0],
    )
    sorted_hash, order = torch.sort(source_hash)
    positions = torch.searchsorted(sorted_hash, reference_hash)
    in_range = positions < sorted_hash.shape[0]
    clamped = positions.clamp_max(sorted_hash.shape[0] - 1)
    present = in_range & (sorted_hash[clamped] == reference_hash)

    output = source.feats.new_zeros(
        (reference.coords.shape[0], *source.feats.shape[1:])
    ) + source.feats.sum() * 0
    if torch.any(present):
        output[present] = source.feats.index_select(0, order[positions[present]])
    return reference.replace(output)


def _empty_mesh(device: torch.device) -> Mesh:
    return Mesh(
        vertices=torch.empty((0, 3), dtype=torch.float32, device=device),
        faces=torch.empty((0, 3), dtype=torch.int32, device=device),
    )


def _mesh_from_flexible_dual_grid(
    vertices: sp.SparseTensor,
    intersected: sp.SparseTensor,
    quad_lerp: sp.SparseTensor,
    resolution: int,
    *,
    train: bool,
) -> Mesh:
    if (
        vertices.coords.shape[0] == 0
        or intersected.coords.shape[0] == 0
        or quad_lerp.coords.shape[0] == 0
    ):
        return _empty_mesh(vertices.device)
    return Mesh(
        *flexible_dual_grid_to_mesh(
            vertices.coords[:, 1:],
            vertices.feats,
            intersected.feats,
            quad_lerp.feats,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=resolution,
            train=train,
        )
    )


def _physical_positions(coords: torch.Tensor, latent_level: int) -> torch.Tensor:
    resolution = latent_level_grid_resolution(latent_level)
    return (coords[:, 1:].float() + 0.5) / resolution - 0.5


def _latent_scales(coords: torch.Tensor, latent_level: int) -> torch.Tensor:
    return torch.full(
        (coords.shape[0],),
        latent_level_spatial_extent(latent_level),
        dtype=torch.float32,
        device=coords.device,
    )


def _to_storage_coords(coords: torch.Tensor, latent_level: int) -> torch.Tensor:
    if coords.shape[0] == 0:
        return coords.new_empty((0, coords.shape[1]))
    storage_xyz = local_coords_to_storage_coords(
        coords[:, 1:].to(torch.int64),
        latent_level,
    )
    return torch.cat(
        [coords[:, :1], storage_xyz.to(dtype=coords.dtype)],
        dim=1,
    ).contiguous()


def _mapping_to_union_sparse_tensor(
    tensors: Mapping[int, sp.SparseTensor],
) -> sp.SparseTensor:
    if len(tensors) == 0:
        raise ValueError("Cannot create a union tensor from an empty mapping")
    union_parts = []
    for latent_level in sorted(int(key) for key in tensors.keys()):
        tensor = tensors[latent_level]
        union_parts.append(
            sp.SparseTensor(
                feats=tensor.feats,
                coords=_to_storage_coords(tensor.coords, latent_level),
                shape=torch.Size([tensor.shape[0], tensor.feats.shape[1]]),
            )
        )
    if len(union_parts) == 1:
        return union_parts[0]
    return sp.sparse_batchwise_cat(union_parts)


def _split_union_by_structure(
    union: sp.SparseTensor,
    structure: Mapping[int, torch.Tensor],
) -> Dict[int, sp.SparseTensor]:
    output = {}
    for latent_level in sorted(int(key) for key in structure.keys()):
        local_coords = structure[latent_level].to(device=union.device, dtype=torch.int32)
        storage_coords = _to_storage_coords(local_coords, latent_level)
        selected = sp.sparse_select_by_coords(union, storage_coords)
        output[latent_level] = build_level_sparse_tensor(
            selected.feats,
            local_coords,
            batch_size=union.shape[0],
            latent_level=latent_level,
        )
    return output


class FlexiDualGridMultiScaleShapeVaeEncoder(nn.Module):
    """Paper Stage II encoder E_z: SparseConv -> CrossAttn -> 8x SelfAttn."""

    def __init__(
        self,
        resolution: int = 1024,
        latent_channels: int = 32,
        model_channels: int = 1024,
        num_frequencies: int = 6,
        lambda_init: float = 1.0,
        num_self_attn_blocks: int = 8,
        num_cross_attn_blocks: int = 1,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        use_checkpoint: bool = False,
        sparse_model_channels: Sequence[int] = (64, 128, 256, 512, 1024),
        sparse_num_blocks: Sequence[int] = (4, 8, 16, 4),
    ):
        super().__init__()
        finest_resolution = latent_level_grid_resolution(LATENT_LEVELS[-1])
        if resolution % 16 != 0 or resolution // 16 != finest_resolution:
            raise ValueError(
                f"The paper SparseConv stem downsamples by 16 to {finest_resolution}; "
                f"got input resolution {resolution}"
            )
        if sparse_model_channels[-1] != model_channels:
            raise ValueError("SparseConv output channels must match transformer model_channels")

        self.resolution = resolution
        self.latent_channels = latent_channels
        self.model_channels = model_channels
        self.latent_levels = LATENT_LEVELS
        self.smpe = ScaleModulatedPositionEmbedder(
            num_frequencies=num_frequencies,
            in_channels=3,
            lambda_init=lambda_init,
        )
        self.feature_encoder = SparseOvoxelFeatureEncoder(
            in_channels=6,
            model_channels=sparse_model_channels,
            num_blocks=sparse_num_blocks,
            mlp_ratio=mlp_ratio,
            use_checkpoint=use_checkpoint,
        )
        self.query_input = sp.SparseLinear(self.smpe.out_channels, model_channels)
        self.cross_blocks = nn.ModuleList([
            SparseCrossAttentionBlock(
                model_channels,
                ctx_channels=model_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qk_rms_norm_cross=qk_rms_norm_cross,
                use_checkpoint=use_checkpoint,
            )
            for _ in range(num_cross_attn_blocks)
        ])
        self.latent_blocks = nn.ModuleList([
            SparseTransformerBlock(
                model_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_mode="full",
                use_checkpoint=use_checkpoint,
                qk_rms_norm=qk_rms_norm,
            )
            for _ in range(num_self_attn_blocks)
        ])
        self.to_latent = sp.SparseLinear(model_channels, 2 * latent_channels)
        self.initialize_weights()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def _build_structure_queries(
        self,
        structure: Mapping[int, torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[int, sp.SparseTensor]:
        expected_levels = set(LATENT_LEVELS)
        actual_levels = {int(key) for key in structure.keys()}
        if actual_levels != expected_levels:
            raise ValueError(
                f"Expected Stage II structure levels {sorted(expected_levels)}, "
                f"got {sorted(actual_levels)}"
            )
        queries = {}
        for latent_level in sorted(int(key) for key in structure.keys()):
            coords = structure[latent_level].to(device=device, dtype=torch.int32)
            positions = _physical_positions(coords, latent_level)
            scales = _latent_scales(coords, latent_level)
            feats = self.smpe(positions, scales).to(dtype=dtype)
            queries[latent_level] = build_level_sparse_tensor(
                feats,
                coords,
                batch_size=batch_size,
                latent_level=latent_level,
            )
        return queries

    def forward(
        self,
        vertices: sp.SparseTensor,
        intersected: sp.SparseTensor,
        structure: Mapping[int, torch.Tensor],
        sample_posterior: bool = False,
        return_raw: bool = False,
    ):
        x = vertices.replace(torch.cat([
            vertices.feats - 0.5,
            intersected.feats.float() - 0.5,
        ], dim=1))
        context = self.feature_encoder(x)
        query_mapping = self._build_structure_queries(
            structure,
            batch_size=vertices.shape[0],
            dtype=context.dtype,
            device=vertices.device,
        )
        queries = _mapping_to_union_sparse_tensor(query_mapping)
        h = self.query_input(queries)
        for block in self.cross_blocks:
            h = block(h, context)
        for block in self.latent_blocks:
            h = block(h)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        stats = self.to_latent(h)
        mean, logvar = stats.feats.chunk(2, dim=-1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z_feats = mean + std * torch.randn_like(std)
        else:
            z_feats = mean

        latents = _split_union_by_structure(stats.replace(z_feats), structure)
        if not return_raw:
            return latents
        means = _split_union_by_structure(stats.replace(mean), structure)
        logvars = _split_union_by_structure(stats.replace(logvar), structure)
        return (
            latents,
            {latent_level: value.feats for latent_level, value in means.items()},
            {latent_level: value.feats for latent_level, value in logvars.items()},
        )


class FlexiDualGridMultiScaleShapeVaeDecoder(nn.Module):
    """Paper Stage II decoder D_z with SMPE-conditioned latent tokens."""

    def __init__(
        self,
        resolution: int = 1024,
        query_resolution: int = 64,
        latent_channels: int = 32,
        model_channels: int = 1024,
        num_frequencies: int = 6,
        lambda_init: float = 1.0,
        num_latent_blocks: int = 8,
        num_query_cross_blocks: int = 1,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        query_mask_threshold: float = 0.0,
        use_gt_query_mask_in_train: bool = True,
        use_checkpoint: bool = False,
        voxel_margin: float = 0.5,
        tail_model_channels: Sequence[int] = (1024, 512, 256, 128, 64),
        tail_num_blocks: Sequence[int] = (4, 16, 8, 4, 0),
        tail_block_type: Sequence[str] = (
            "SparseConvNeXtBlock3d",
            "SparseConvNeXtBlock3d",
            "SparseConvNeXtBlock3d",
            "SparseConvNeXtBlock3d",
            "SparseConvNeXtBlock3d",
        ),
        tail_up_block_type: Sequence[str] = (
            "SparseResBlockC2S3d",
            "SparseResBlockC2S3d",
            "SparseResBlockC2S3d",
            "SparseResBlockC2S3d",
        ),
        tail_block_args: Optional[Sequence[Dict[str, object]]] = None,
    ):
        super().__init__()
        if resolution % query_resolution != 0:
            raise ValueError("resolution must be divisible by query_resolution")
        upsample_ratio = resolution // query_resolution
        if upsample_ratio <= 0 or (upsample_ratio & (upsample_ratio - 1)) != 0:
            raise ValueError("resolution / query_resolution must be a positive power of two")
        if query_resolution != latent_level_grid_resolution(LATENT_LEVELS[-1]):
            raise ValueError("The paper decoder uses a 64^3 dense query grid")
        expected_tail_stages = int(math.log2(upsample_ratio)) + 1
        if len(tail_model_channels) != expected_tail_stages:
            raise ValueError(
                f"Expected {expected_tail_stages} tail stages, got {len(tail_model_channels)}"
            )
        if tail_model_channels[0] != model_channels:
            raise ValueError("The first tail channel width must match model_channels")
        if tail_block_args is None:
            tail_block_args = tuple(
                {"use_checkpoint": use_checkpoint}
                for _ in tail_model_channels
            )

        self.resolution = resolution
        self.query_resolution = query_resolution
        self.voxel_margin = voxel_margin
        self.query_mask_threshold = query_mask_threshold
        self.use_gt_query_mask_in_train = use_gt_query_mask_in_train

        self.smpe = ScaleModulatedPositionEmbedder(
            num_frequencies=num_frequencies,
            in_channels=3,
            lambda_init=lambda_init,
        )
        self.query_pos_embedder = FourierPositionEmbedder(
            num_frequencies=num_frequencies,
            in_channels=3,
        )
        self.latent_input = sp.SparseLinear(
            latent_channels + self.smpe.out_channels,
            model_channels,
        )
        self.latent_blocks = nn.ModuleList([
            SparseTransformerBlock(
                model_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_mode="full",
                use_checkpoint=use_checkpoint,
                qk_rms_norm=qk_rms_norm,
            )
            for _ in range(num_latent_blocks)
        ])
        self.query_input = sp.SparseLinear(
            self.query_pos_embedder.out_channels,
            model_channels,
        )
        self.query_blocks = nn.ModuleList([
            SparseCrossAttentionBlock(
                model_channels,
                ctx_channels=model_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qk_rms_norm_cross=qk_rms_norm_cross,
                use_checkpoint=use_checkpoint,
            )
            for _ in range(num_query_cross_blocks)
        ])
        self.query_mask_head = sp.SparseLinear(model_channels, 1)
        self.query_to_tail = sp.SparseLinear(model_channels, tail_model_channels[0])
        self.tail = SparseQueryDecoderTail(
            7,
            list(tail_model_channels),
            list(tail_num_blocks),
            list(tail_block_type),
            list(tail_up_block_type),
            list(tail_block_args),
            pred_subdiv=True,
        )
        self.tail_up_block_type = list(tail_up_block_type)
        self.gt_cache_downsamples = nn.ModuleList([
            self._make_teacher_forcing_downsample(block_name)
            for block_name in reversed(self.tail_up_block_type)
        ])
        self.initialize_weights()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def set_resolution(self, resolution: int) -> None:
        if resolution % self.query_resolution != 0:
            raise ValueError("resolution must be divisible by query_resolution")
        upsample_ratio = resolution // self.query_resolution
        expected_tail_stages = int(math.log2(upsample_ratio)) + 1
        if expected_tail_stages != len(self.tail.model_channels):
            raise ValueError("The existing tail does not match the requested resolution")
        self.resolution = resolution

    @staticmethod
    def _make_teacher_forcing_downsample(up_block_name: str) -> nn.Module:
        if up_block_name == "SparseResBlockC2S3d":
            return sp.SparseSpatial2Channel(2)
        raise ValueError(f"Unsupported upsample block: {up_block_name}")

    def _prepare_latent_tokens(
        self,
        latents: Mapping[int, sp.SparseTensor],
    ) -> sp.SparseTensor:
        expected_levels = set(LATENT_LEVELS)
        actual_levels = {int(key) for key in latents.keys()}
        if actual_levels != expected_levels:
            raise ValueError(
                f"Expected Stage II latent levels {sorted(expected_levels)}, "
                f"got {sorted(actual_levels)}"
            )
        token_mapping = {}
        for latent_level in sorted(int(key) for key in latents.keys()):
            latent = latents[latent_level]
            positions = _physical_positions(latent.coords, latent_level)
            scales = _latent_scales(latent.coords, latent_level)
            smpe = self.smpe(positions, scales).to(dtype=latent.feats.dtype)
            token_mapping[latent_level] = latent.replace(
                torch.cat([latent.feats, smpe], dim=-1)
            )
        union = _mapping_to_union_sparse_tensor(token_mapping)
        h = self.latent_input(union)
        for block in self.latent_blocks:
            h = block(h)
        return h

    def _expand_level_query_coords(
        self,
        latent: sp.SparseTensor,
        latent_level: int,
    ) -> torch.Tensor:
        level_resolution = latent_level_grid_resolution(latent_level)
        if self.query_resolution % level_resolution != 0:
            raise ValueError(
                f"query_resolution={self.query_resolution} is incompatible with "
                f"latent-level resolution {level_resolution}"
            )
        factor = self.query_resolution // level_resolution
        if latent.coords.shape[0] == 0 or factor == 1:
            return latent.coords

        offsets = torch.stack(
            torch.meshgrid(
                *[
                    torch.arange(factor, dtype=torch.int32, device=latent.device)
                    for _ in range(3)
                ],
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 3)
        base = latent.coords[:, 1:].to(torch.int32) * factor
        xyz = (base[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        batch_idx = latent.coords[:, :1].repeat_interleave(offsets.shape[0], dim=0)
        return torch.cat([batch_idx, xyz], dim=1).contiguous()

    def _build_query_tensor(
        self,
        latents: Mapping[int, sp.SparseTensor],
        dtype: torch.dtype,
    ) -> sp.SparseTensor:
        query_coords = []
        batch_size = None
        for latent_level in sorted(int(key) for key in latents.keys()):
            latent = latents[latent_level]
            batch_size = latent.shape[0] if batch_size is None else batch_size
            query_coords.append(self._expand_level_query_coords(latent, latent_level))
        if batch_size is None:
            raise ValueError("Cannot build queries from an empty latent mapping")

        coords = torch.cat(query_coords, dim=0)
        if coords.shape[0] > 0:
            coords = torch.unique(coords, dim=0, sorted=True)
            positions = (coords[:, 1:].float() + 0.5) / self.query_resolution - 0.5
            feats = self.query_pos_embedder(positions).to(dtype=dtype)
        else:
            coords = coords.contiguous()
            feats = torch.empty(
                (0, self.query_pos_embedder.out_channels),
                dtype=dtype,
                device=coords.device,
            )
        query = sp.SparseTensor(
            feats=feats,
            coords=coords,
            shape=torch.Size([batch_size, feats.shape[1]]),
        )
        return self.query_input(query)

    def _build_gt_query_tensor(self, gt_intersected: sp.SparseTensor) -> sp.SparseTensor:
        occupancy = gt_intersected.feats.new_ones(
            (gt_intersected.feats.shape[0], 1),
            dtype=torch.float32,
        )
        x = gt_intersected.replace(occupancy)
        for downsample in self.gt_cache_downsamples:
            x = downsample(x)
        return x

    def _select_query_features(
        self,
        query_h: sp.SparseTensor,
        mask_logits: sp.SparseTensor,
        gt_query: Optional[sp.SparseTensor],
    ) -> Tuple[sp.SparseTensor, Optional[torch.Tensor]]:
        query_target = None
        if gt_query is not None:
            query_target = _sparse_coords_membership_mask(
                mask_logits.coords,
                gt_query.coords,
            )

        if self.training and self.use_gt_query_mask_in_train and gt_query is not None:
            support_mask = _sparse_coords_membership_mask(
                gt_query.coords,
                query_h.coords,
            )
            gt_query = sp.sparse_select_by_rows(gt_query, support_mask)
            selected = sp.sparse_select_by_coords(query_h, gt_query.coords)
            selected = self.query_to_tail(selected)
            return gt_query.replace(selected.feats), query_target

        keep_mask = mask_logits.feats[:, 0] > self.query_mask_threshold
        selected = sp.sparse_select_by_rows(query_h, keep_mask)
        return self.query_to_tail(selected), query_target

    def _decode_mesh_training(
        self,
        h: sp.SparseTensor,
        gt_intersected: sp.SparseTensor,
    ):
        h = _align_sparse_to_reference(h, gt_intersected)
        vertices = h.replace(
            (1 + 2 * self.voxel_margin) * torch.sigmoid(h.feats[..., 0:3])
            - self.voxel_margin
        )
        intersected_logits = h.replace(h.feats[..., 3:6])
        quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
        mesh = [
            _mesh_from_flexible_dual_grid(
                vertex,
                intersection,
                split,
                self.resolution,
                train=True,
            )
            for vertex, intersection, split in zip(
                vertices,
                gt_intersected,
                quad_lerp,
            )
        ]
        return mesh, vertices, intersected_logits

    def _decode_mesh_eval(self, h: sp.SparseTensor) -> List[Mesh]:
        if h.coords.shape[0] == 0:
            return [_empty_mesh(h.device) for _ in range(h.shape[0])]
        vertices = h.replace(
            (1 + 2 * self.voxel_margin) * torch.sigmoid(h.feats[..., 0:3])
            - self.voxel_margin
        )
        intersected = h.replace(h.feats[..., 3:6] > 0)
        quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
        return [
            _mesh_from_flexible_dual_grid(
                vertex,
                intersection,
                split,
                self.resolution,
                train=False,
            )
            for vertex, intersection, split in zip(vertices, intersected, quad_lerp)
        ]

    def forward(
        self,
        latents: Mapping[int, sp.SparseTensor],
        gt_intersected: Optional[sp.SparseTensor] = None,
        return_aux: bool = False,
    ):
        latent_h = self._prepare_latent_tokens(latents)
        query_h = self._build_query_tensor(latents, latent_h.dtype)
        for block in self.query_blocks:
            query_h = block(query_h, latent_h)
        mask_logits = self.query_mask_head(query_h)

        gt_query = (
            self._build_gt_query_tensor(gt_intersected)
            if gt_intersected is not None
            else None
        )
        tail_input, query_target = self._select_query_features(
            query_h,
            mask_logits,
            gt_query,
        )
        decoded = self.tail(tail_input)

        if self.training:
            if gt_intersected is None:
                raise ValueError("gt_intersected is required during decoder training")
            h, subs_gt, subs = decoded
            mesh, vertices, intersected_logits = self._decode_mesh_training(
                h,
                gt_intersected,
            )
            return (
                mesh,
                vertices,
                intersected_logits,
                subs_gt,
                subs,
                mask_logits,
                query_target,
            )

        h = decoded[0] if isinstance(decoded, tuple) else decoded
        mesh = self._decode_mesh_eval(h)
        if return_aux:
            return mesh, mask_logits
        return mesh
