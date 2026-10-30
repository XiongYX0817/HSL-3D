from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from . import rembg, samplers
from .base import Pipeline
from ..modules import image_feature_extractor
from ..modules import sparse as sp
from ..representations import Mesh
from ..utils.multiscale_shape_latent_utils import (
    LATENT_LEVELS,
    denormalize_multiscale_sparse_tensor,
    local_coords_to_storage_coords,
    split_union_sparse_tensor_by_level,
    latent_level_to_varlen,
)
from ..utils.octree_utils import (
    batch_octree_frontiers,
    decode_octree_logits,
    rebuild_octree,
)


class HSL3DImageTo3DPipeline(Pipeline):
    """Two-stage image-to-geometry pipeline described by HSL-3D."""

    model_names_to_load = (
        "octree_structure_flow_model",
        "octree_structure_decoder",
        "multiscale_shape_flow_model",
        "multiscale_shape_decoder",
    )

    def __init__(
        self,
        models: Optional[Dict[str, nn.Module]] = None,
        stage1_sampler: Optional[samplers.Sampler] = None,
        stage2_sampler: Optional[samplers.Sampler] = None,
        stage1_sampler_params: Optional[dict] = None,
        stage2_sampler_params: Optional[dict] = None,
        stage1_normalization: Optional[dict] = None,
        stage2_normalization: Optional[dict] = None,
        image_cond_model=None,
        rembg_model=None,
        low_vram: bool = True,
        stage1_cond_resolution: int = 512,
        stage2_cond_resolution: int = 1024,
        octree_base_resolution: int = 16,
        octree_num_levels: int = 3,
        structure_threshold: float = 0.0,
        max_num_tokens: int = 32768,
    ):
        if models is None:
            return
        super().__init__(models)
        self.stage1_sampler = stage1_sampler
        self.stage2_sampler = stage2_sampler
        self.stage1_sampler_params = stage1_sampler_params or {}
        self.stage2_sampler_params = stage2_sampler_params or {}
        self.stage1_normalization = stage1_normalization
        self.stage2_normalization = stage2_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = bool(low_vram)
        self.stage1_cond_resolution = int(stage1_cond_resolution)
        self.stage2_cond_resolution = int(stage2_cond_resolution)
        self.octree_base_resolution = int(octree_base_resolution)
        self.octree_num_levels = int(octree_num_levels)
        self.structure_threshold = float(structure_threshold)
        self.max_num_tokens = int(max_num_tokens)
        self._device = torch.device("cpu")

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config_file: str = "pipeline.json",
    ) -> "HSL3DImageTo3DPipeline":
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        stage1_sampler_cfg = args["stage1_sampler"]
        pipeline.stage1_sampler = getattr(
            samplers,
            stage1_sampler_cfg["name"],
        )(**stage1_sampler_cfg.get("args", {}))
        pipeline.stage1_sampler_params = dict(stage1_sampler_cfg.get("params", {}))

        stage2_sampler_cfg = args["stage2_sampler"]
        pipeline.stage2_sampler = getattr(
            samplers,
            stage2_sampler_cfg["name"],
        )(**stage2_sampler_cfg.get("args", {}))
        pipeline.stage2_sampler_params = dict(stage2_sampler_cfg.get("params", {}))

        pipeline.stage1_normalization = args.get("stage1_normalization")
        pipeline.stage2_normalization = args.get("stage2_normalization")
        image_cond_cfg = args["image_cond_model"]
        pipeline.image_cond_model = getattr(
            image_feature_extractor,
            image_cond_cfg["name"],
        )(**image_cond_cfg.get("args", {}))

        rembg_cfg = args.get("rembg_model")
        pipeline.rembg_model = None
        if rembg_cfg is not None:
            pipeline.rembg_model = getattr(rembg, rembg_cfg["name"])(
                **rembg_cfg.get("args", {})
            )

        pipeline.low_vram = bool(args.get("low_vram", True))
        pipeline.stage1_cond_resolution = int(args.get("stage1_cond_resolution", 512))
        pipeline.stage2_cond_resolution = int(args.get("stage2_cond_resolution", 1024))
        pipeline.octree_base_resolution = int(args.get("octree_base_resolution", 16))
        pipeline.octree_num_levels = int(args.get("octree_num_levels", 3))
        pipeline.structure_threshold = float(args.get("structure_threshold", 0.0))
        pipeline.max_num_tokens = int(args.get("max_num_tokens", 32768))
        pipeline._device = torch.device("cpu")
        return pipeline

    def to(self, device) -> "HSL3DImageTo3DPipeline":
        self._device = torch.device(device)
        if not self.low_vram:
            for model in self.models.values():
                model.to(self._device)
            self.image_cond_model.to(self._device)
            if self.rembg_model is not None:
                self.rembg_model.to(self._device)
        return self

    @staticmethod
    def _has_alpha(image: Image.Image) -> bool:
        if image.mode != "RGBA":
            return False
        alpha = np.asarray(image)[:, :, 3]
        return not np.all(alpha == 255)

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Remove the background, crop the foreground, and premultiply alpha."""
        image = image.copy()
        max_size = max(image.size)
        scale = min(1.0, 1024.0 / max_size)
        if scale < 1.0:
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.Resampling.LANCZOS,
            )

        if not self._has_alpha(image):
            if self.rembg_model is None:
                raise ValueError(
                    "Image preprocessing requires an alpha channel or a configured rembg model"
                )
            if self.low_vram:
                self.rembg_model.to(self.device)
            image = self.rembg_model(image.convert("RGB"))
            if self.low_vram:
                self.rembg_model.cpu()

        rgba = np.asarray(image.convert("RGBA"))
        alpha = rgba[:, :, 3]
        foreground = np.argwhere(alpha > int(0.8 * 255))
        if foreground.shape[0] == 0:
            raise ValueError("No foreground was found after image preprocessing")
        x_min, x_max = foreground[:, 1].min(), foreground[:, 1].max()
        y_min, y_max = foreground[:, 0].min(), foreground[:, 0].max()
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        size = max(1, int(max(x_max - x_min, y_max - y_min)))
        crop_box = (
            center_x - size // 2,
            center_y - size // 2,
            center_x + size // 2,
            center_y + size // 2,
        )
        rgba = np.asarray(image.crop(crop_box).convert("RGBA"), dtype=np.float32) / 255.0
        rgb = rgba[:, :, :3] * rgba[:, :, 3:4]
        return Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")

    def get_condition(
        self,
        image: Image.Image,
        resolution: int,
        *,
        num_samples: int,
    ) -> dict:
        self.image_cond_model.image_size = int(resolution)
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model([image])
        if self.low_vram:
            self.image_cond_model.cpu()
        if num_samples > 1:
            cond = cond.repeat(num_samples, 1, 1)
        return {"cond": cond, "neg_cond": torch.zeros_like(cond)}

    @staticmethod
    def _denormalize_dense_latent(
        latent: torch.Tensor,
        normalization: Optional[dict],
    ) -> torch.Tensor:
        if normalization is None:
            return latent
        mean = torch.as_tensor(
            normalization["mean"],
            dtype=latent.dtype,
            device=latent.device,
        ).reshape(1, -1, 1, 1, 1)
        std = torch.as_tensor(
            normalization["std"],
            dtype=latent.dtype,
            device=latent.device,
        ).reshape(1, -1, 1, 1, 1)
        return latent * std + mean

    def sample_stage1(
        self,
        cond: dict,
        *,
        num_samples: int,
        sampler_params: Optional[dict],
        generator: torch.Generator,
        structure_threshold: float,
    ):
        flow_model = self.models["octree_structure_flow_model"]
        noise = torch.randn(
            (
                num_samples,
                flow_model.in_channels,
                flow_model.resolution,
                flow_model.resolution,
                flow_model.resolution,
            ),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        effective_params = {
            **self.stage1_sampler_params,
            **(sampler_params or {}),
        }
        if self.low_vram:
            flow_model.to(self.device)
        stage1_latent = self.stage1_sampler.sample(
            flow_model,
            noise,
            **cond,
            **effective_params,
            verbose=True,
            tqdm_desc="Sampling Stage I octree latents",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        decoder_input = self._denormalize_dense_latent(
            stage1_latent,
            self.stage1_normalization,
        )
        decoder = self.models["octree_structure_decoder"]
        if self.low_vram:
            decoder.to(self.device)
        logits = decoder(decoder_input)
        if self.low_vram:
            decoder.cpu()

        octree_infos = decode_octree_logits(
            logits,
            threshold=structure_threshold,
            num_levels=self.octree_num_levels,
        )
        octrees = [
            rebuild_octree(
                octree_info,
                base_resolution=self.octree_base_resolution,
                num_levels=self.octree_num_levels,
            )
            for octree_info in octree_infos
        ]
        frontier = batch_octree_frontiers(
            octrees,
            num_levels=self.octree_num_levels,
        )
        return stage1_latent, octree_infos, octrees, frontier

    def _validate_token_counts(
        self,
        frontier: Mapping[int, torch.Tensor],
        *,
        num_samples: int,
        max_num_tokens: int,
    ) -> None:
        counts = torch.zeros(num_samples, dtype=torch.long)
        for coords in frontier.values():
            if coords.shape[0] == 0:
                continue
            counts += torch.bincount(coords[:, 0].cpu(), minlength=num_samples)
        if torch.any(counts == 0):
            bad = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
            raise ValueError(f"Stage I produced empty octrees for samples {bad}")
        if torch.any(counts > max_num_tokens):
            bad = {
                int(index): int(counts[index])
                for index in torch.nonzero(
                    counts > max_num_tokens,
                    as_tuple=False,
                ).flatten().tolist()
            }
            raise ValueError(
                f"Stage II token limit exceeded (max_num_tokens={max_num_tokens}): {bad}"
            )

    def build_stage2_noise(
        self,
        frontier: Mapping[int, torch.Tensor],
        *,
        num_samples: int,
        generator: torch.Generator,
    ) -> Tuple[sp.SparseTensor, sp.VarLenTensor]:
        flow_model = self.models["multiscale_shape_flow_model"]
        coords_parts = []
        level_parts = []
        layout = []
        level_slices = {level: [] for level in LATENT_LEVELS}
        start = 0
        for batch_idx in range(num_samples):
            batch_start = start
            for latent_level in LATENT_LEVELS:
                level_coords = frontier[latent_level]
                level_coords = level_coords[level_coords[:, 0] == batch_idx]
                local_coords = level_coords[:, 1:]
                if local_coords.shape[0] == 0:
                    continue
                storage_coords = local_coords_to_storage_coords(
                    local_coords,
                    latent_level,
                ).to(dtype=torch.int32)
                batched_coords = torch.cat(
                    [
                        torch.full(
                            (storage_coords.shape[0], 1),
                            batch_idx,
                            dtype=torch.int32,
                        ),
                        storage_coords,
                    ],
                    dim=1,
                )
                coords_parts.append(batched_coords)
                level_parts.append(
                    torch.full(
                        (storage_coords.shape[0],),
                        latent_level,
                        dtype=torch.long,
                    )
                )
                segment = slice(start, start + storage_coords.shape[0])
                level_slices[latent_level].append(segment)
                start = segment.stop
            layout.append(slice(batch_start, start))

        coords = torch.cat(coords_parts, dim=0).to(self.device)
        latent_levels = torch.cat(level_parts, dim=0).to(self.device)
        noise = sp.SparseTensor(
            feats=torch.randn(
                coords.shape[0],
                flow_model.in_channels,
                generator=generator,
                device=self.device,
            ),
            coords=coords,
            shape=torch.Size([num_samples, flow_model.in_channels]),
        )
        noise.register_spatial_cache("layout", layout)
        latent_level = latent_level_to_varlen(latent_levels, layout)
        compact_slices = {
            level: tuple(slices)
            for level, slices in level_slices.items()
            if slices
        }
        noise.register_spatial_cache("multiscale_level_slices", compact_slices)
        latent_level._cache["multiscale_level_slices"] = compact_slices
        return noise, latent_level

    def sample_stage2(
        self,
        cond: dict,
        frontier: Mapping[int, torch.Tensor],
        *,
        num_samples: int,
        sampler_params: Optional[dict],
        generator: torch.Generator,
    ):
        flow_model = self.models["multiscale_shape_flow_model"]
        noise, latent_level = self.build_stage2_noise(
            frontier,
            num_samples=num_samples,
            generator=generator,
        )
        effective_params = {
            **self.stage2_sampler_params,
            **(sampler_params or {}),
        }
        if self.low_vram:
            flow_model.to(self.device)
        multiscale_shape_latent = self.stage2_sampler.sample(
            flow_model,
            noise,
            **cond,
            latent_level=latent_level,
            **effective_params,
            verbose=True,
            tqdm_desc="Sampling Stage II shape latents",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        if self.stage2_normalization is not None:
            multiscale_shape_latent = denormalize_multiscale_sparse_tensor(
                multiscale_shape_latent,
                latent_level,
                self.stage2_normalization,
            )
        latents_by_level = split_union_sparse_tensor_by_level(
            multiscale_shape_latent,
            latent_level,
        )
        return multiscale_shape_latent, latent_level, latents_by_level

    def decode_stage2(
        self,
        latents_by_level: Mapping[int, sp.SparseTensor],
    ) -> List[Mesh]:
        decoder = self.models["multiscale_shape_decoder"]
        if self.low_vram:
            decoder.to(self.device)
        meshes = decoder(latents_by_level)
        if self.low_vram:
            decoder.cpu()
        return meshes

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        stage1_sampler_params: Optional[dict] = None,
        stage2_sampler_params: Optional[dict] = None,
        preprocess_image: bool = True,
        return_latent: bool = False,
        max_num_tokens: Optional[int] = None,
        structure_threshold: Optional[float] = None,
    ):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        token_limit = self.max_num_tokens if max_num_tokens is None else int(max_num_tokens)
        if token_limit <= 0 or token_limit > self.max_num_tokens:
            raise ValueError(
                f"max_num_tokens must be in [1, {self.max_num_tokens}], got {token_limit}"
            )
        threshold = (
            self.structure_threshold
            if structure_threshold is None
            else float(structure_threshold)
        )
        if preprocess_image:
            image = self.preprocess_image(image)
        elif not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        stage1_cond = self.get_condition(
            image,
            self.stage1_cond_resolution,
            num_samples=num_samples,
        )
        stage1_latent, octree_infos, octrees, frontier = self.sample_stage1(
            stage1_cond,
            num_samples=num_samples,
            sampler_params=stage1_sampler_params,
            generator=generator,
            structure_threshold=threshold,
        )
        del stage1_cond
        self._validate_token_counts(
            frontier,
            num_samples=num_samples,
            max_num_tokens=token_limit,
        )
        stage2_cond = self.get_condition(
            image,
            self.stage2_cond_resolution,
            num_samples=num_samples,
        )
        multiscale_shape_latent, latent_level, latents_by_level = self.sample_stage2(
            stage2_cond,
            frontier,
            num_samples=num_samples,
            sampler_params=stage2_sampler_params,
            generator=generator,
        )
        meshes = self.decode_stage2(latents_by_level)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        if not return_latent:
            return meshes
        return meshes, {
            "stage1_latent": stage1_latent,
            "octree_info": octree_infos,
            "octree": octrees,
            "octree_frontier": frontier,
            "multiscale_shape_latent": multiscale_shape_latent,
            "multiscale_shape_latents_by_level": latents_by_level,
            "latent_level": latent_level,
        }
