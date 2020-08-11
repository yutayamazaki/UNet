import torch.nn as nn
import torchvision

from models import fpn, unet_resnet


def load_model(
    num_classes: int, architecture: str, backbone: str = 'resnet18',
    pretrained: bool = True
):
    """Load segmentation model which has specified backbone.
    Args:
        num_classes (int): Number of classes.
        architecture (str): 'unet' or 'fpn'.
        backbone (str): What ResNet model used as backbone.
    """
    architecture = architecture.lower()
    if architecture == 'unet':
        return unet_resnet.UNetResNet(num_classes, backbone, pretrained)
    elif architecture == 'fpn':
        return fpn.FPN(num_classes, backbone, pretrained)
    else:
        raise ValueError('Get invalid architecture name.')


def load_resnet_backbone(
    backbone: str = 'resnet18', pretrained: bool = True
) -> nn.Module:
    if backbone == 'resnet18':
        return torchvision.models.resnet18(pretrained)
    elif backbone == 'resnet34':
        return torchvision.models.resnet34(pretrained)
    elif backbone == 'resnet50':
        return torchvision.models.resnet50(pretrained)
    elif backbone == 'resnet101':
        return torchvision.models.resnet101(pretrained)
    elif backbone == 'resnet152':
        return torchvision.models.resnet152(pretrained)
    elif backbone == 'resnext50_32x4d':
        return torchvision.models.resnext50_32x4d(pretrained)
    elif backbone == 'resnext101_32x8d':
        return torchvision.models.resnext101_32x8d(pretrained)
    elif backbone == 'wide_resnet50_2':
        return torchvision.models.wide_resnet50_2(pretrained)
    elif backbone == 'wide_resnet101_2':
        return torchvision.models.wide_resnet101_2(pretrained)
    else:
        attributes = (
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2',
            'wide_resnet101_2'
        )
        raise ValueError(f'backbone must be in {attributes}.')
