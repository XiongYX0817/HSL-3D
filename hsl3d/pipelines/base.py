import json
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from .. import models as model_registry


class Pipeline:
    """Base class for config-driven inference pipelines."""

    model_names_to_load: Optional[tuple[str, ...]] = None

    def __init__(self, models: Optional[Dict[str, nn.Module]] = None):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config_file: str = "pipeline.json",
    ) -> "Pipeline":
        """Load a local model bundle or a Hugging Face model repository."""
        local_config = os.path.join(path, config_file)
        is_local = os.path.isfile(local_config)
        if is_local:
            resolved_config = local_config
        else:
            from huggingface_hub import hf_hub_download

            resolved_config = hf_hub_download(path, config_file)

        with open(resolved_config, "r", encoding="utf-8") as stream:
            args = json.load(stream)["args"]

        loaded_models = {}
        for name, model_path in args["models"].items():
            if cls.model_names_to_load is not None and name not in cls.model_names_to_load:
                continue
            candidate = f"{path}/{model_path}" if not is_local else model_path
            if is_local and not os.path.isabs(model_path):
                local_candidate = os.path.join(path, model_path)
                if (
                    os.path.isfile(f"{local_candidate}.json")
                    and os.path.isfile(f"{local_candidate}.safetensors")
                ):
                    candidate = local_candidate
            loaded_models[name] = model_registry.from_pretrained(candidate)

        pipeline = cls(loaded_models)
        pipeline._pretrained_args = args
        pipeline._pretrained_path = path
        return pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, "_device"):
            return torch.device(self._device)
        for model in self.models.values():
            if hasattr(model, "device"):
                return torch.device(model.device)
        for model in self.models.values():
            try:
                return next(model.parameters()).device
            except (AttributeError, StopIteration):
                continue
        raise RuntimeError("No model device is available")

    def to(self, device) -> "Pipeline":
        device = torch.device(device)
        for model in self.models.values():
            model.to(device)
        self._device = device
        return self

    def cuda(self) -> "Pipeline":
        return self.to(torch.device("cuda"))

    def cpu(self) -> "Pipeline":
        return self.to(torch.device("cpu"))
