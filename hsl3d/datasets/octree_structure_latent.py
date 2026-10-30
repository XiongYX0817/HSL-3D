import os
import json
from typing import Optional, Union
import numpy as np
import torch
from .components import StandardDatasetBase, ImageConditionedMixin
from .. import models


class OctreeStructureLatentVisualizationMixin:
    def __init__(
        self,
        *args,
        pretrained_octree_decoder: Optional[str] = None,
        octree_decoder_path: Optional[str] = None,
        octree_decoder_checkpoint: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.octree_decoder = None
        self.pretrained_octree_decoder = pretrained_octree_decoder
        self.octree_decoder_path = octree_decoder_path
        self.octree_decoder_checkpoint = octree_decoder_checkpoint
        
    def _load_octree_decoder(self):
        if self.octree_decoder is not None:
            return
        if self.octree_decoder_path is not None:
            cfg = json.load(open(os.path.join(self.octree_decoder_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(
                self.octree_decoder_path,
                'ckpts',
                f'decoder_{self.octree_decoder_checkpoint}.pt',
            )
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        elif self.pretrained_octree_decoder is not None:
            decoder = models.from_pretrained(self.pretrained_octree_decoder)
        else:
            raise ValueError(
                "Either octree_decoder_path or pretrained_octree_decoder is required for visualization"
            )
        self.octree_decoder = decoder.cuda().eval()

    def _delete_octree_decoder(self):
        self.octree_decoder = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        self._load_octree_decoder()
        oc = []
        if self.normalization:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            oc.append(self.octree_decoder(z[i:i+batch_size]))
        oc = torch.cat(oc, dim=0)
        self._delete_octree_decoder()
        return oc

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict]):
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0']
        # x_0 = self.decode_latent(x_0.cuda())
        
        # renderer = VoxelRenderer()
        # renderer.rendering_options.resolution = 512
        # renderer.rendering_options.ssaa = 4
        
        # # build camera
        # yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
        # yaw_offset = -16 / 180 * np.pi
        # yaw = [y + yaw_offset for y in yaw]
        # pitch = [20 / 180 * np.pi for _ in range(4)]
        # exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

        images = []
        
        # Build each representation
        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            # coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            # resolution = x_0.shape[-1]
            # color = coords / resolution
            # rep = Voxel(
            #     origin=[-0.5, -0.5, -0.5],
            #     voxel_size=1/resolution,
            #     coords=coords,
            #     attrs=color,
            #     layout={
            #         'color': slice(0, 3),
            #     }
            # )
            image = torch.zeros(3, 1024, 1024).cuda()
            # tile = [2, 2]
            # for j, (ext, intr) in enumerate(zip(exts, ints)):
            #     res = renderer.render(rep, ext, intr, colors_overwrite=color)
            #     image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            
        return torch.stack(images)


class OctreeStructureLatent(OctreeStructureLatentVisualizationMixin, StandardDatasetBase):
    """
    Octree structure latent dataset
    
    Args:
        roots (str): path to the dataset
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_octree_decoder (str): optional pretrained decoder reference
        octree_decoder_path (str): optional decoder training directory
        octree_decoder_checkpoint (str): decoder checkpoint tag
    """
    def __init__(self,
        roots: str,
        *,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        pretrained_octree_decoder: Optional[str] = None,
        octree_decoder_path: Optional[str] = None,
        octree_decoder_checkpoint: Optional[str] = None,
    ):
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_octree_decoder=pretrained_octree_decoder,
            octree_decoder_path=octree_decoder_path,
            octree_decoder_checkpoint=octree_decoder_checkpoint,
        )
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
  
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata['octree_latent_encoded'] == True]
        stats['With latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats
                
    def get_instance(self, root, instance):
        with np.load(os.path.join(root['octree_latent'], f'{instance}.npz')) as latent:
            z = torch.tensor(latent['z']).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        pack = {
            'x_0': z,
            "instance": instance,
        }
        return pack


class ImageConditionedOctreeStructureLatent(ImageConditionedMixin, OctreeStructureLatent):
    """
    Image-conditioned octree structure dataset
    """
    pass
    
