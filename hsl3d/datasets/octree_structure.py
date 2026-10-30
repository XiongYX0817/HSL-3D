import os
import torch
from .components import StandardDatasetBase
from ..utils.octree_utils import octree2tensor


class OctreeStructure(StandardDatasetBase):
    """
    Octree structure dataset

    Args:
        roots (str): path to the dataset
        level (int): number of levels in the octree structure
        resolution (int): resolution of the coarest octree level
        min_aesthetic_score (float): minimum aesthetic score of the instances to be included in the dataset
    """

    def __init__(self,
        roots,
        level: int = 3,
        resolution: int = 16,
        min_aesthetic_score: float = 5.0,
    ):
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.value_range = (0, 1)
        self.level = level

        super().__init__(roots)

    def filter_metadata(self, metadata):
        def model_path_exists(sha256: str) -> bool:
            for root in self.roots:
                if os.path.exists(os.path.join(root, f'octrees_{self.resolution}', f'{sha256}.pt')):
                    return True
            return False
        stats = {}
        mask = metadata['sha256'].apply(model_path_exists)
        metadata = metadata[mask]
        stats["Octreelized"] = len(metadata)
        # metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        # stats[f'Aesthetic Score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        octree_info = torch.load(os.path.join(root, f'octrees_{self.resolution}', f'{instance}.pt'), map_location="cpu")
        octree = dict()
        cur_resolution = self.resolution
        for l in range(self.level):
            coords = octree_info[l].int().contiguous()
            ss = torch.zeros(1, cur_resolution, cur_resolution, cur_resolution, dtype=torch.long)
            ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
            octree[l] = ss
            cur_resolution *= 2
        # return {"octree": octree2tensor(octree)}
        return {"octree": octree2tensor(octree), "instance": instance}

    @torch.no_grad()
    def visualize_sample(self, data: dict):
        images = torch.zeros(1, 3, 1024, 1024).cuda()
        return images
