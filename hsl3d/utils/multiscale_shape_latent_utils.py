from __future__ import annotations

from fractions import Fraction
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch

from ..modules import sparse as sp


# Zero-based latent levels correspond to the paper octree levels O1, O2, and O3.
# Level 0 is the coarsest representation and level 2 is the finest.
LATENT_LEVELS = (0, 1, 2)
LATENT_LEVEL_TO_GRID_RESOLUTION = {
    0: 16,
    1: 32,
    2: 64,
}
MAX_LATENT_GRID_RESOLUTION = max(LATENT_LEVEL_TO_GRID_RESOLUTION.values())

PACK_FORMAT = "multiscale_shape_latent"
PACK_LEVEL_FORMAT = "zero_based_coarse_to_fine"
LEVEL_STATS_FORMAT = "multiscale_shape_latent_level_stats"


def validate_latent_level(latent_level: int) -> int:
    latent_level = int(latent_level)
    if latent_level not in LATENT_LEVEL_TO_GRID_RESOLUTION:
        raise ValueError(f"Unsupported latent_level: {latent_level}")
    return latent_level


def latent_level_grid_resolution(latent_level: int) -> int:
    return LATENT_LEVEL_TO_GRID_RESOLUTION[validate_latent_level(latent_level)]


def latent_level_spatial_extent(latent_level: int) -> float:
    return 1.0 / latent_level_grid_resolution(latent_level)


def latent_level_storage_stride(latent_level: int) -> int:
    resolution = latent_level_grid_resolution(latent_level)
    return MAX_LATENT_GRID_RESOLUTION // resolution


def latent_level_sparse_scale(
    latent_level: int,
) -> Tuple[Fraction, Fraction, Fraction]:
    scale = Fraction(latent_level_storage_stride(latent_level), 1)
    return (scale, scale, scale)


def local_coords_to_storage_coords(
    local_coords: torch.Tensor,
    latent_level: int,
) -> torch.Tensor:
    if local_coords.ndim != 2 or local_coords.shape[1] != 3:
        raise ValueError(
            f"Expected local_coords to have shape [N, 3], got {tuple(local_coords.shape)}"
        )
    if local_coords.dtype not in [torch.int32, torch.int64]:
        raise ValueError(f"Expected integer local_coords, got {local_coords.dtype}")
    stride = latent_level_storage_stride(latent_level)
    return ((local_coords.to(torch.int64) * 2) + 1) * stride


def storage_coords_to_local_coords(
    storage_coords: torch.Tensor,
    latent_level: int,
) -> torch.Tensor:
    if storage_coords.ndim != 2 or storage_coords.shape[1] != 3:
        raise ValueError(
            f"Expected storage_coords to have shape [N, 3], got {tuple(storage_coords.shape)}"
        )
    if storage_coords.dtype not in [torch.int32, torch.int64]:
        raise ValueError(f"Expected integer storage_coords, got {storage_coords.dtype}")
    stride = latent_level_storage_stride(latent_level)
    storage_coords = storage_coords.to(torch.int64)
    if storage_coords.numel() == 0:
        return storage_coords.to(dtype=torch.int32)
    if torch.any(storage_coords % stride != 0):
        raise ValueError(
            f"storage_coords are not divisible by stride={stride} "
            f"for latent_level={latent_level}"
        )
    reduced = storage_coords // stride
    if torch.any((reduced - 1) % 2 != 0):
        raise ValueError(
            "storage_coords do not match the expected odd-grid pattern "
            f"for latent_level={latent_level}"
        )
    return ((reduced - 1) // 2).to(dtype=torch.int32)


def latent_level_tensor_to_levels(
    latent_level: Union[sp.VarLenTensor, torch.Tensor],
) -> torch.Tensor:
    if isinstance(latent_level, sp.VarLenTensor):
        latent_level = latent_level.feats
    if latent_level.ndim == 2 and latent_level.shape[1] == 1:
        latent_level = latent_level[:, 0]
    if latent_level.ndim != 1:
        raise ValueError(
            f"Expected latent_level to flatten to shape [T], got {tuple(latent_level.shape)}"
        )
    levels = latent_level.to(dtype=torch.long)
    if levels.numel() > 0:
        valid = torch.zeros_like(levels, dtype=torch.bool)
        for level in LATENT_LEVELS:
            valid |= levels == level
        if not torch.all(valid):
            bad = torch.unique(levels[~valid]).tolist()
            raise ValueError(f"Unsupported latent levels: {bad}")
    return levels


def latent_level_to_varlen(
    latent_level: torch.Tensor,
    layout: Sequence[slice],
) -> sp.VarLenTensor:
    if latent_level.ndim != 1:
        raise ValueError(
            f"Expected flat latent_level tensor, got {tuple(latent_level.shape)}"
        )
    return sp.VarLenTensor(latent_level[:, None], list(layout))


def parse_level_normalization_stats(
    normalization: Optional[Mapping[str, object]],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Optional[Dict[int, Dict[str, torch.Tensor]]]:
    if normalization is None:
        return None
    if str(normalization.get("format")) != LEVEL_STATS_FORMAT:
        raise ValueError(
            f"Unsupported normalization format: {normalization.get('format')}"
        )
    if str(normalization.get("level_format")) != PACK_LEVEL_FORMAT:
        raise ValueError(
            f"Unsupported level format: {normalization.get('level_format')}"
        )
    raw_levels = normalization.get("levels")
    if not isinstance(raw_levels, Mapping):
        raise ValueError("normalization['levels'] must be a mapping")

    parsed: Dict[int, Dict[str, torch.Tensor]] = {}
    for raw_level, payload in raw_levels.items():
        latent_level = validate_latent_level(int(raw_level))
        if not isinstance(payload, Mapping):
            raise ValueError(f"levels[{raw_level!r}] must be a mapping")
        expected_resolution = latent_level_grid_resolution(latent_level)
        resolution = int(payload.get("grid_resolution", -1))
        if resolution != expected_resolution:
            raise ValueError(
                f"levels[{raw_level!r}] grid_resolution must be "
                f"{expected_resolution}, got {resolution}"
            )
        mean = payload.get("mean")
        std = payload.get("std")
        if mean is None or std is None:
            raise ValueError(f"Missing mean/std for latent_level={latent_level}")
        parsed[latent_level] = {
            "mean": torch.as_tensor(mean, device=device, dtype=dtype).reshape(1, -1),
            "std": torch.as_tensor(std, device=device, dtype=dtype).reshape(1, -1),
        }

    if set(parsed) != set(LATENT_LEVELS):
        raise ValueError(
            f"Normalization must contain levels {list(LATENT_LEVELS)}, "
            f"got {sorted(parsed)}"
        )
    return parsed


def normalize_levelwise_feats(
    feats: torch.Tensor,
    latent_level: Union[sp.VarLenTensor, torch.Tensor],
    normalization: Optional[Mapping[str, object]],
) -> torch.Tensor:
    stats = parse_level_normalization_stats(
        normalization,
        device=feats.device,
        dtype=feats.dtype,
    )
    if stats is None or feats.shape[0] == 0:
        return feats
    levels = latent_level_tensor_to_levels(latent_level).to(device=feats.device)
    out = feats.clone()
    for level in LATENT_LEVELS:
        mask = levels == level
        if torch.any(mask):
            out[mask] = (out[mask] - stats[level]["mean"]) / stats[level]["std"]
    return out


def denormalize_levelwise_feats(
    feats: torch.Tensor,
    latent_level: Union[sp.VarLenTensor, torch.Tensor],
    normalization: Optional[Mapping[str, object]],
) -> torch.Tensor:
    stats = parse_level_normalization_stats(
        normalization,
        device=feats.device,
        dtype=feats.dtype,
    )
    if stats is None or feats.shape[0] == 0:
        return feats
    levels = latent_level_tensor_to_levels(latent_level).to(device=feats.device)
    out = feats.clone()
    for level in LATENT_LEVELS:
        mask = levels == level
        if torch.any(mask):
            out[mask] = out[mask] * stats[level]["std"] + stats[level]["mean"]
    return out


def normalize_multiscale_sparse_tensor(
    x: sp.SparseTensor,
    latent_level: Union[sp.VarLenTensor, torch.Tensor],
    normalization: Optional[Mapping[str, object]],
) -> sp.SparseTensor:
    return x.replace(
        normalize_levelwise_feats(x.feats, latent_level, normalization)
    )


def denormalize_multiscale_sparse_tensor(
    x: sp.SparseTensor,
    latent_level: Union[sp.VarLenTensor, torch.Tensor],
    normalization: Optional[Mapping[str, object]],
) -> sp.SparseTensor:
    return x.replace(
        denormalize_levelwise_feats(x.feats, latent_level, normalization)
    )


def build_level_sparse_tensor(
    feats: torch.Tensor,
    coords: torch.Tensor,
    *,
    batch_size: int,
    latent_level: int,
) -> sp.SparseTensor:
    latent_level = validate_latent_level(latent_level)
    return sp.SparseTensor(
        feats=feats,
        coords=coords,
        shape=torch.Size([batch_size, feats.shape[1]]),
        scale=latent_level_sparse_scale(latent_level),
    )


def split_union_sparse_tensor_by_level(
    x: sp.SparseTensor,
    latent_level: Union[sp.VarLenTensor, torch.Tensor],
) -> Dict[int, sp.SparseTensor]:
    levels = latent_level_tensor_to_levels(latent_level).to(device=x.device)
    if levels.shape[0] != x.coords.shape[0]:
        raise ValueError(
            f"latent_level length mismatch: got {levels.shape[0]}, "
            f"expected {x.coords.shape[0]}"
        )

    output: Dict[int, sp.SparseTensor] = {}
    for level in LATENT_LEVELS:
        rows = torch.nonzero(levels == level, as_tuple=False).squeeze(1)
        selected = sp.sparse_select_by_rows(x, rows)
        local_coords = selected.coords.clone()
        if local_coords.shape[0] > 0:
            local_coords[:, 1:] = storage_coords_to_local_coords(
                local_coords[:, 1:],
                level,
            )
        output[level] = build_level_sparse_tensor(
            selected.feats,
            local_coords,
            batch_size=x.shape[0],
            latent_level=level,
        )
    return output
