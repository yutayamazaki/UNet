from typing import Optional

import torch.nn as nn

from .unet import UNet
from .unet_resnet import UNetResNet


def load_unet(backbone: Optional[str], num_classes: int) -> nn.Module:
    """Load specified segmentation model."""
    if backbone is None:
        return UNet(in_channels=3, num_classes=num_classes)
    else:
        return UNetResNet(
            num_classes=num_classes, backbone=backbone, pretrained=True
        )
