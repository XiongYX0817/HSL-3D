import importlib

__attributes = {
    # Stage I
    'OctreeStructure': 'octree_structure',
    'OctreeStructureLatent': 'octree_structure_latent',
    'ImageConditionedOctreeStructureLatent': 'octree_structure_latent',

    # Stage II flow matching
    'MultiScaleShapeLatentDataset': 'multiscale_shape_latent',
    'ImageConditionedMultiScaleShapeLatentDataset': 'multiscale_shape_latent',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':    
    from .octree_structure import OctreeStructure
    from .octree_structure_latent import OctreeStructureLatent, ImageConditionedOctreeStructureLatent
    from .multiscale_shape_latent import (
        MultiScaleShapeLatentDataset,
        ImageConditionedMultiScaleShapeLatentDataset,
    )
    
