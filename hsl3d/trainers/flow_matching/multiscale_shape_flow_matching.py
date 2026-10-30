from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from ...utils.multiscale_shape_latent_utils import (
    LATENT_LEVELS,
    latent_level_tensor_to_levels,
)
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import ImageConditionedMixin, MultiImageConditionedMixin
from .mixins.text_conditioned import TextConditionedMixin
from .sparse_flow_matching import SparseFlowMatchingTrainer


class MultiScaleShapeFlowMatchingTrainer(SparseFlowMatchingTrainer):
    """
    Sparse flow-matching trainer with level-balanced multiscale loss aggregation.
    """

    def training_losses(
        self,
        x_0,
        latent_level,
        cond=None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)

        pred = self.training_models["denoiser"](
            x_t,
            t * 1000,
            cond,
            latent_level=latent_level,
            **kwargs,
        )
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)

        latent_levels = latent_level_tensor_to_levels(latent_level).to(device=x_0.device)
        sample_losses = []
        level_loss_terms = {level: [] for level in LATENT_LEVELS}
        for i in range(x_0.shape[0]):
            sample_slice = x_0.layout[i]
            sample_pred = pred.feats[sample_slice]
            sample_target = target.feats[sample_slice]
            sample_levels = latent_levels[sample_slice]
            sample_level_losses = []
            for level in LATENT_LEVELS:
                level_mask = sample_levels == level
                if not torch.any(level_mask):
                    continue
                level_loss = F.mse_loss(sample_pred[level_mask], sample_target[level_mask])
                sample_level_losses.append(level_loss)
                level_loss_terms[level].append(level_loss)
            if len(sample_level_losses) == 0:
                sample_losses.append(sample_pred.sum() * 0)
            else:
                sample_losses.append(torch.stack(sample_level_losses).mean())

        terms = edict()
        terms["mse"] = torch.stack(sample_losses).mean()
        terms["loss"] = terms["mse"]
        for level, losses in level_loss_terms.items():
            if len(losses) == 0:
                continue
            terms[f"level{level}_mse"] = torch.stack(losses).mean()

        mse_per_instance = torch.stack(sample_losses).detach().cpu().numpy()
        time_bin = np.digitize(t.detach().cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}


class MultiScaleShapeFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, MultiScaleShapeFlowMatchingTrainer):
    """
    Multiscale sparse flow-matching trainer with classifier-free guidance.
    """

    pass


class TextConditionedMultiScaleShapeFlowMatchingCFGTrainer(TextConditionedMixin, MultiScaleShapeFlowMatchingCFGTrainer):
    """
    Text-conditioned multiscale sparse flow-matching trainer with classifier-free guidance.
    """

    pass


class ImageConditionedMultiScaleShapeFlowMatchingCFGTrainer(ImageConditionedMixin, MultiScaleShapeFlowMatchingCFGTrainer):
    """
    Image-conditioned multiscale sparse flow-matching trainer with classifier-free guidance.
    """

    pass


class MultiImageConditionedMultiScaleShapeFlowMatchingCFGTrainer(MultiImageConditionedMixin, MultiScaleShapeFlowMatchingCFGTrainer):
    """
    Multi-image-conditioned multiscale sparse flow-matching trainer with classifier-free guidance.
    """

    pass
