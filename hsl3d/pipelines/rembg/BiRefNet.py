import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class BiRefNet:
    def __init__(self, model_name: str = "briaai/RMBG-2.0"):
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model.eval()
        self.transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to(self, device):
        self.model.to(device)
        return self

    def cuda(self):
        return self.to(torch.device("cuda"))

    def cpu(self):
        return self.to(torch.device("cpu"))

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.copy()
        image_size = image.size
        input_image = self.transform_image(image).unsqueeze(0).to(self.device)
        prediction = self.model(input_image)[-1].sigmoid().cpu()[0].squeeze()
        mask = transforms.ToPILImage()(prediction).resize(image_size)
        image.putalpha(mask)
        return image
