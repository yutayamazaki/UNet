import torch.nn as nn
import torchvision


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


def conv3x3(in_channels: int, out_channels: int, rate=1):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rate,
                  padding=rate, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )
    return conv
