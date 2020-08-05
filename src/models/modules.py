import torch.nn as nn
import torchvision


def load_resnet_backbone(
    backbone: str = 'resnet18', pretrained: bool = True
) -> torchvision.models.ResNet:
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
    else:
        raise ValueError('backbone must be resnet18, 34, 50, 101, 152.')


def conv3x3(in_channels: int, out_channels: int, rate=1):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rate,
                  padding=rate, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )
    return conv
