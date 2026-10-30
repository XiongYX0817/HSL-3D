import importlib


__attributes = {
    "HSL3DImageTo3DPipeline": "hsl3d_image_to_3d",
}

__submodules = ["samplers", "rembg"]
__all__ = list(__attributes) + __submodules


def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module = importlib.import_module(f".{__attributes[name]}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, config_file: str = "pipeline.json"):
    import json
    import os

    local_config = os.path.join(path, config_file)
    if os.path.isfile(local_config):
        resolved_config = local_config
    else:
        from huggingface_hub import hf_hub_download

        resolved_config = hf_hub_download(path, config_file)
    with open(resolved_config, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    return __getattr__(config["name"]).from_pretrained(path, config_file=config_file)


if __name__ == "__main__":
    from . import rembg, samplers
    from .hsl3d_image_to_3d import HSL3DImageTo3DPipeline
