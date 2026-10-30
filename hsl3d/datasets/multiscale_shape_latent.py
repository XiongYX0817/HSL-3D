import json
import os
from typing import Mapping, Optional, Union

import numpy as np
import torch

from .. import models
from ..modules.sparse import SparseTensor, VarLenTensor
from ..utils.multiscale_shape_latent_utils import (
    LATENT_LEVELS,
    PACK_FORMAT,
    PACK_LEVEL_FORMAT,
    denormalize_multiscale_sparse_tensor,
    local_coords_to_storage_coords,
    parse_level_normalization_stats,
    normalize_levelwise_feats,
    split_union_sparse_tensor_by_level,
    latent_level_to_varlen,
)
from ..utils.render_utils import get_renderer, yaw_pitch_r_fov_to_extrinsics_intrinsics
from .components import ImageConditionedMixin, StandardDatasetBase
from .shape_latent import ShapeLatentVisualizationMixin
from ..utils.data_utils import load_balanced_group_indices


def _load_multiscale_shape_latent_pack(root_dir: str, instance: str, storage_format: Optional[str]) -> Mapping[str, object]:
    candidates = []
    if storage_format == "npy":
        candidates.append(os.path.join(root_dir, f"{instance}.npy"))
    elif storage_format == "npz":
        candidates.append(os.path.join(root_dir, f"{instance}.npz"))
    else:
        candidates.extend([
            os.path.join(root_dir, f"{instance}.npz"),
            os.path.join(root_dir, f"{instance}.npy"),
        ])

    latent_path = None
    for candidate in candidates:
        if os.path.exists(candidate):
            latent_path = candidate
            break
    if latent_path is None:
        raise FileNotFoundError(f"Latent pack not found for {instance}. Tried: {candidates}")

    ext = os.path.splitext(latent_path)[1].lower()
    if ext == ".npz":
        data = np.load(latent_path, allow_pickle=False)
        pack = {k: data[k] for k in data.files}
    elif ext == ".npy":
        pack = np.load(latent_path, allow_pickle=True).item()
    else:
        raise ValueError(f"Unsupported latent extension: {ext}")

    normalized = {}
    for key, value in pack.items():
        if isinstance(value, np.ndarray) and value.ndim == 0:
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized


def _validate_pack(pack: Mapping[str, object]) -> None:
    if str(pack.get("format")) != PACK_FORMAT:
        raise ValueError(f"Unsupported latent format: {pack.get('format')}")
    if str(pack.get("level_format")) != PACK_LEVEL_FORMAT:
        raise ValueError(f"Unsupported level format: {pack.get('level_format')}")
    coords = np.asarray(pack["coords"])
    feats = np.asarray(pack["feats"])
    levels = np.asarray(pack["level"])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape [N, 3], got {coords.shape}")
    if feats.ndim != 2:
        raise ValueError(f"feats must have shape [N, C], got {feats.shape}")
    if levels.ndim != 1:
        raise ValueError(f"level must have shape [N], got {levels.shape}")
    if not (coords.shape[0] == feats.shape[0] == levels.shape[0]):
        raise ValueError("coords, feats, and level must have matching first dimension")
    unknown_levels = np.setdiff1d(np.unique(levels), np.asarray(LATENT_LEVELS))
    if unknown_levels.size > 0:
        raise ValueError(f"Unexpected latent levels: {unknown_levels.tolist()}")


class MultiScaleShapeLatentVisualizationMixin(ShapeLatentVisualizationMixin):
    def _load_shape_decoder(self):
        if self.shape_decoder is not None:
            return
        if self.shape_decoder_path is not None:
            cfg = json.load(open(os.path.join(self.shape_decoder_path, "config.json"), "r"))
            decoder = getattr(models, cfg["models"]["decoder"]["name"])(**cfg["models"]["decoder"]["args"])
            ckpt_path = os.path.join(
                self.shape_decoder_path,
                "ckpts",
                f"decoder_{self.shape_decoder_checkpoint}.pt",
            )
            decoder.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        elif self.pretrained_shape_decoder is not None:
            decoder = models.from_pretrained(self.pretrained_shape_decoder)
        else:
            raise ValueError(
                "Either shape_decoder_path or pretrained_shape_decoder must be provided for visualization"
            )
        decoder.set_resolution(self.resolution)
        self.shape_decoder = decoder.cuda().eval()

    @torch.no_grad()
    def decode_latent(self, z: SparseTensor, latent_level: VarLenTensor, batch_size: int = 4):
        self._load_shape_decoder()
        reps = []
        for i in range(0, z.shape[0], batch_size):
            z_chunk = z[i:i + batch_size]
            level_chunk = latent_level[i:i + batch_size]
            if self.normalization is not None:
                z_chunk = denormalize_multiscale_sparse_tensor(
                    z_chunk,
                    level_chunk,
                    self.normalization,
                )
            latents_by_level = split_union_sparse_tensor_by_level(z_chunk, level_chunk)
            reps.extend(self.shape_decoder(latents_by_level))
        self._delete_shape_decoder()
        return reps

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[dict, SparseTensor]):
        if isinstance(x_0, SparseTensor):
            raise ValueError(
                "MultiScaleShapeLatentDataset.visualize_sample expects x_0 and latent_level"
            )

        z = x_0["x_0"]
        latent_level = x_0["latent_level"]
        reps = self.decode_latent(z.cuda(), latent_level.cuda())

        yaw = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaw_offset = -16 / 180 * np.pi
        yaw = [y + yaw_offset for y in yaw]
        pitch = [20 / 180 * np.pi for _ in range(4)]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

        renderer = get_renderer(reps[0])
        images = []
        for representation in reps:
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res["normal"]
            images.append(image)
        return torch.stack(images)


class MultiScaleShapeLatentDataset(MultiScaleShapeLatentVisualizationMixin, StandardDatasetBase):
    """
    Multiscale structured latent dataset for shape generation.
    """

    def __init__(
        self,
        roots: str,
        *,
        resolution: int,
        min_aesthetic_score: float = 5.0,
        max_tokens: int = 32768,
        latent_key: str = "shape_latent",
        normalization: Optional[dict] = None,
        pretrained_shape_decoder: Optional[str] = None,
        shape_decoder_path: Optional[str] = None,
        shape_decoder_checkpoint: Optional[str] = None,
    ):
        self.normalization = normalization
        self.min_aesthetic_score = min_aesthetic_score
        self.max_tokens = max_tokens
        self.latent_key = latent_key
        self.resolution = resolution
        self.value_range = (0, 1)
        parse_level_normalization_stats(normalization)

        super().__init__(
            roots,
            pretrained_shape_decoder=pretrained_shape_decoder,
            shape_decoder_path=shape_decoder_path,
            shape_decoder_checkpoint=shape_decoder_checkpoint,
        )

        self.loads = [self.metadata.loc[sha256, f"{latent_key}_tokens"] for _, sha256 in self.instances]

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f"{self.latent_key}_encoded"] == True]
        stats["With latent"] = len(metadata)
        metadata = metadata[metadata["aesthetic_score"] >= self.min_aesthetic_score]
        stats[f"Aesthetic score >= {self.min_aesthetic_score}"] = len(metadata)
        metadata = metadata[metadata[f"{self.latent_key}_tokens"] <= self.max_tokens]
        stats[f"Num tokens <= {self.max_tokens}"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        storage_format = None
        if "shape_latent_storage" in self.metadata.columns:
            maybe_format = self.metadata.loc[instance, "shape_latent_storage"]
            if isinstance(maybe_format, str):
                storage_format = maybe_format

        pack = _load_multiscale_shape_latent_pack(root[self.latent_key], instance, storage_format)
        _validate_pack(pack)

        coords = torch.tensor(np.asarray(pack["coords"]), dtype=torch.int32)
        feats = torch.tensor(np.asarray(pack["feats"]), dtype=torch.float32)
        latent_level = torch.tensor(np.asarray(pack["level"]), dtype=torch.long)
        storage_coords = torch.empty_like(coords)
        for level in LATENT_LEVELS:
            mask = latent_level == level
            if not torch.any(mask):
                continue
            storage_coords[mask] = local_coords_to_storage_coords(
                coords[mask],
                level,
            ).to(dtype=torch.int32)

        if self.normalization is not None:
            feats = normalize_levelwise_feats(
                feats,
                latent_level_to_varlen(latent_level, [slice(0, coords.shape[0])]),
                self.normalization,
            )

        return {
            "coords": storage_coords,
            "feats": feats,
            "latent_level": latent_level,
        }

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b["coords"].shape[0] for b in batch], split_size)

        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            coords = []
            feats = []
            latent_levels = []
            layout = []
            start = 0
            for i, b in enumerate(sub_batch):
                n = b["coords"].shape[0]
                coords.append(torch.cat([torch.full((n, 1), i, dtype=torch.int32), b["coords"]], dim=-1))
                feats.append(b["feats"])
                latent_levels.append(b["latent_level"])
                layout.append(slice(start, start + n))
                start += n

            coords = torch.cat(coords, dim=0)
            feats = torch.cat(feats, dim=0)
            latent_levels = torch.cat(latent_levels, dim=0)

            pack = {}
            pack["x_0"] = SparseTensor(
                coords=coords,
                feats=feats,
            )
            pack["x_0"]._shape = torch.Size([len(group), *sub_batch[0]["feats"].shape[1:]])
            pack["x_0"].register_spatial_cache("layout", layout)
            pack["latent_level"] = latent_level_to_varlen(latent_levels, layout)

            keys = [k for k in sub_batch[0].keys() if k not in ["coords", "feats", "latent_level"]]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]

            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs


class ImageConditionedMultiScaleShapeLatentDataset(
    ImageConditionedMixin,
    MultiScaleShapeLatentDataset,
):
    """
    Image-conditioned multiscale structured latent dataset for shape generation.
    """

    pass
