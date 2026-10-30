from typing import Optional


class ShapeLatentVisualizationMixin:
    """Shared lazy decoder lifecycle for shape-latent visualization datasets."""

    def __init__(
        self,
        *args,
        pretrained_shape_decoder: Optional[str] = None,
        shape_decoder_path: Optional[str] = None,
        shape_decoder_checkpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shape_decoder = None
        self.pretrained_shape_decoder = pretrained_shape_decoder
        self.shape_decoder_path = shape_decoder_path
        self.shape_decoder_checkpoint = shape_decoder_checkpoint

    def _delete_shape_decoder(self) -> None:
        self.shape_decoder = None
