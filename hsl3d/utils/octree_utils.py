from typing import Dict, List, Mapping, Optional

import torch

def voxel_upsample(x, scale):
    if scale == 1:
        return x
    return x.repeat_interleave(scale, dim=-1).repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-3)

def octree2tensor(octree):
    level = len(octree.keys())
    tensors = []
    for i in range(level):
        tensors.append(voxel_upsample(octree[i], 2**(level-i-1)))
    return torch.cat(tensors, dim=-4)


def _coords_membership(source: torch.Tensor, query: torch.Tensor, factor: int) -> torch.Tensor:
    if source.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.bool, device=source.device)
    if query.shape[0] == 0:
        return torch.zeros((source.shape[0],), dtype=torch.bool, device=source.device)
    source_key = source[:, 0] * factor**2 + source[:, 1] * factor + source[:, 2]
    query_key = query[:, 0] * factor**2 + query[:, 1] * factor + query[:, 2]
    return torch.isin(source_key, query_key)


def decode_octree_logits(
    logits: Mapping[int, torch.Tensor],
    *,
    threshold: float = 0.0,
    num_levels: Optional[int] = None,
) -> List[Dict[int, torch.Tensor]]:
    """Convert batched Stage I decoder logits to per-sample octree-info coords.

    Coarse levels are split masks; the finest level is an occupancy mask.
    The returned coordinates stay on CPU so they can be rebuilt and validated
    without retaining dense GPU tensors.
    """
    if num_levels is None:
        num_levels = len(logits)
    if num_levels <= 0:
        raise ValueError("num_levels must be positive")
    missing = [level for level in range(num_levels) if level not in logits]
    if missing:
        raise ValueError(f"Missing octree logit levels: {missing}")

    batch_size = int(logits[0].shape[0])
    output = [dict() for _ in range(batch_size)]
    for level in range(num_levels):
        level_logits = logits[level]
        if level_logits.ndim != 5 or level_logits.shape[1] != 1:
            raise ValueError(
                f"logits[{level}] must have shape [B, 1, R, R, R], "
                f"got {tuple(level_logits.shape)}"
            )
        if level_logits.shape[0] != batch_size:
            raise ValueError("All octree logit levels must share the same batch size")
        occupied = level_logits[:, 0] > threshold
        for batch_idx in range(batch_size):
            output[batch_idx][level] = (
                occupied[batch_idx]
                .nonzero(as_tuple=False)
                .to(dtype=torch.int32, device="cpu")
                .contiguous()
            )
    return output


def rebuild_octree(
    octree_info: Mapping[int, torch.Tensor],
    *,
    base_resolution: int = 16,
    num_levels: int = 3,
) -> Dict[int, torch.Tensor]:
    """Apply the training-time top-down split semantics to obtain leaf voxels."""
    if base_resolution <= 0:
        raise ValueError("base_resolution must be positive")
    if num_levels <= 0:
        raise ValueError("num_levels must be positive")
    missing = [level for level in range(num_levels) if level not in octree_info]
    if missing:
        raise ValueError(f"Missing octree-info levels: {missing}")

    finest_resolution = base_resolution * 2 ** (num_levels - 1)
    finest_coords = torch.as_tensor(
        octree_info[num_levels - 1],
        dtype=torch.int32,
    ).reshape(-1, 3)
    factor = finest_resolution + 1
    leaves: Dict[int, torch.Tensor] = {}

    for level in range(num_levels - 1):
        level_resolution = base_resolution * 2**level
        scale = finest_resolution // level_resolution
        parents = torch.div(finest_coords, scale, rounding_mode="floor")
        occupied_parents = torch.unique(parents, dim=0, sorted=True)
        split_coords = torch.as_tensor(
            octree_info[level],
            dtype=torch.int32,
            device=occupied_parents.device,
        ).reshape(-1, 3)
        split_mask = _coords_membership(occupied_parents, split_coords, factor)
        level_leaves = occupied_parents[~split_mask].contiguous()
        leaves[level] = level_leaves.cpu()

        leaf_mask = _coords_membership(parents, level_leaves, factor)
        finest_coords = finest_coords[~leaf_mask].contiguous()

    leaves[num_levels - 1] = finest_coords.cpu().contiguous()
    return leaves


def octree_to_frontier(
    octree: Mapping[int, torch.Tensor],
    *,
    batch_idx: int = 0,
    num_levels: int = 3,
) -> Dict[int, torch.Tensor]:
    """Convert octree leaves to zero-based Stage II latent-level frontiers."""
    frontier = {}
    for level in range(num_levels):
        coords = torch.as_tensor(octree.get(level, []), dtype=torch.int32).reshape(-1, 3)
        if coords.shape[0] == 0:
            frontier[level] = torch.empty((0, 4), dtype=torch.int32)
        else:
            frontier[level] = torch.cat(
                [
                    torch.full((coords.shape[0], 1), batch_idx, dtype=torch.int32),
                    coords,
                ],
                dim=1,
            ).contiguous()
    return frontier


def batch_octree_frontiers(
    octrees: List[Mapping[int, torch.Tensor]],
    *,
    num_levels: int = 3,
) -> Dict[int, torch.Tensor]:
    """Combine per-sample octrees into batch-contiguous Stage II frontiers."""
    if len(octrees) == 0:
        raise ValueError("At least one octree is required")
    parts = {level: [] for level in range(num_levels)}
    for batch_idx, octree in enumerate(octrees):
        frontier = octree_to_frontier(
            octree,
            batch_idx=batch_idx,
            num_levels=num_levels,
        )
        for latent_level, coords in frontier.items():
            parts[latent_level].append(coords)
    return {
        latent_level: torch.cat(coords, dim=0).contiguous()
        for latent_level, coords in parts.items()
    }
