from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
    Normalize,
    RandomResizedCrop
)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OwlvitTransform(object):
    def __init__(self, device, image_size=768):
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None] * 255.
        self.mean = self.mean.to(device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None] * 255.
        self.std = self.std.to(device)
        self.device = device
        self.image_size = image_size

    @torch.no_grad()
    def __call__(self, image):
        pixel_values = torch.from_numpy(np.asarray(image))[None, ...]
        pixel_values = pixel_values.permute(0, 3, 1, 2)
        pixel_values = pixel_values.to(self.device).float()
        pixel_values = F.interpolate(pixel_values, (self.image_size, self.image_size), mode="bilinear")
        pixel_values.sub_(self.mean).div_(self.std)
        return pixel_values[0]
    

def build_owlvit_vision_transform(device, is_train: bool = False, image_size=768):

    if is_train:
        resize = RandomResizedCrop((image_size, image_size))
    else:
        resize = Resize((image_size, image_size))

    transform = Compose([
        ToTensor(),
        resize.to(device),
        Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ).to(device)
    ])

    if not is_train:
        return OwlvitTransform(device)
    
    return transform

