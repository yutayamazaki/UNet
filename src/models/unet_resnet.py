from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FPA(nn.Module):
    """ Feature Pyramid Attention module proposed in
        https://arxiv.org/abs/1805.10180
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(FPA, self).__init__()
        self.glob = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.down2_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=2,
                      padding=2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )
        self.down2_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.down3_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )
        self.down3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x_glob = self.glob(x)
        x_glob = F.interpolate(
            x_glob, scale_factor=16, mode='bilinear', align_corners=True
        )
        d2 = self.down2_1(x)
        d3 = self.down3_1(d2)
        d2 = self.down2_2(d2)
        d3 = self.down3_2(d3)

        d3 = F.interpolate(
            d3, scale_factor=2, mode='bilinear', align_corners=True
        )
        d2 = d2 + d3

        d2 = F.interpolate(
            d2, scale_factor=2, mode='bilinear', align_corners=True
        )

        x = self.conv1(x)
        x = x * d2
        x = x + x_glob
        return x


class SSE(nn.Module):

    """ Implementation of 'Channel Squeeze and spatial excitation; SSE'.
    Note:
        https://arxiv.org/abs/1709.01507
    """

    def __init__(self, in_channels: int):
        super(SSE, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        """ [B, C, H, W] -> [B, C, H, W] """
        sq = self.squeeze(x)
        return x * torch.sigmoid(sq)


class CSE(nn.Module):

    """ Implementation of 'Spatial squeeze and Channel Excitation; CSE'.
    Note:
        https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super(CSE, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels, in_channels // reduction, kernel_size=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels // reduction, in_channels, kernel_size=1, stride=1
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """ [B, C, H, W] -> [B, C, H, W] """
        sq = self.gap(x)
        sq = self.relu(self.conv1(sq))
        sq_scaled = torch.sigmoid(self.conv2(sq))
        return x * sq_scaled


class SCSE(nn.Module):

    """ Implementation of SCSE module.
    Note:
        https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels: int):
        super(SCSE, self).__init__()
        self.spatial_att = SSE(in_channels)
        self.channel_att = CSE(in_channels)

    def forward(self, x):
        """ [B, C, H, W] -> [B, C, H, W] """
        return self.spatial_att(x) + self.channel_att(x)


def conv3x3(input_dim: int, output_dim: int, rate: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(
            input_dim, output_dim, kernel_size=3, dilation=rate,
            padding=rate, bias=False
        ),
        nn.BatchNorm2d(output_dim),
        nn.ReLU(True)
    )


class UpBlock(nn.Module):

    def __init__(self, in_channels: int, channels: int, out_channels: int):
        super(UpBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, channels)
        self.conv2 = conv3x3(channels, out_channels)
        self.s_att = SSE(out_channels)
        self.c_att = CSE(out_channels, 16)

    def forward(self, x):
        """ [B, C, H, W] -> [B, C, H*2, W*2] """
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=True
        )
        x = self.conv1(x)
        x = self.conv2(x)
        s = self.s_att(x)
        c = self.c_att(x)
        output = s + c
        return output


class UpBlockv2(nn.Module):

    def __init__(self, up_in: int, x_in: int, n_out: int):
        super(UpBlockv2, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1, bias=False)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.s_att = SSE(n_out)
        self.c_att = CSE(n_out, 16)

    def forward(self, x, e):
        """ [B, C, H, W] -> [B, C, H*2, W*2] """
        x = self.tr_conv(x)
        e = self.x_conv(e)

        cat = torch.cat([x, e], 1)
        cat = self.relu(self.bn(cat))
        s = self.s_att(cat)
        c = self.c_att(cat)
        return s + c


def _load_resnet_backbone(
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


class UNetResNet(nn.Module):
    """ UNet with ResNet encoder. You can use following models as a backbone.

        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2',
        'wide_resnet101_2'

        Difference from original UNet.
            - Concate decoded feature maps to detect objects which has varios
              scales.
            - Add SCSE module after each encode layers.
            - Add Feature Pyramid Attention module.
    """

    def __init__(
        self, num_classes: int, backbone: str = 'resnet18',
        pretrained: bool = True
    ):
        super(UNetResNet, self).__init__()
        resnet: nn.Module = _load_resnet_backbone(backbone, pretrained)

        # Number of feature maps
        num_channels: List[int] = [256, 512, 1024, 2048]
        if backbone in ('resnet18', 'resnet34'):
            num_channels = [64, 128, 256, 512]

        self.backbone = backbone

        self.conv1 = nn.Sequential(
            resnet.conv1,  # type: ignore
            resnet.bn1,  # type: ignore
            resnet.relu  # type: ignore
        )

        self.encode2 = nn.Sequential(
            resnet.layer1, SCSE(in_channels=num_channels[0])  # type: ignore
        )
        self.encode3 = nn.Sequential(
            resnet.layer2, SCSE(in_channels=num_channels[1])  # type: ignore
        )
        self.encode4 = nn.Sequential(
            resnet.layer3, SCSE(in_channels=num_channels[2])  # type: ignore
        )
        self.encode5 = nn.Sequential(
            resnet.layer4, SCSE(in_channels=num_channels[3])  # type: ignore
        )

        self.center = nn.Sequential(
            FPA(in_channels=num_channels[3], out_channels=num_channels[2]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decode5 = UpBlockv2(num_channels[2], num_channels[3], 64)
        self.decode4 = UpBlockv2(64, num_channels[2], 64)
        self.decode3 = UpBlockv2(64, num_channels[1], 64)
        self.decode2 = UpBlockv2(64, num_channels[0], 64)
        self.decode1 = UpBlock(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, num_classes, kernel_size=1, bias=False)
        )

    def forward(self, x):
        """ (B, C, H, W) -> (B, num_classes, H, W) """
        x = self.conv1(x)  # (B, 64, H/2, W/2)
        e2 = self.encode2(x)  # (B, 256, H/2, W/2)
        e3 = self.encode3(e2)  # (B, 512, H/4, W/4)
        e4 = self.encode4(e3)  # (B, 1024, H/8, W/8)
        e5 = self.encode5(e4)  # (B, 2048, H/16, W/16)

        f = self.center(e5)  # (B, 1024, H/32, W/32)

        d5 = self.decode5(f, e5)  # [B, 64, H/16, W/16]
        d4 = self.decode4(d5, e4)  # [B, 64, H/8, W/8]
        d3 = self.decode3(d4, e3)  # [B, 64, H/4, W/4]
        d2 = self.decode2(d3, e2)  # [B, 64, H/2, W/2]
        d1 = self.decode1(d2)  # [B, 256, H, W]

        # Concate feature maps with multiple resolutions to detect objects of
        # various scales.
        f = torch.cat(
            (
                d1,
                F.interpolate(d2, scale_factor=2, mode='bilinear',
                              align_corners=True),
                F.interpolate(d3, scale_factor=4, mode='bilinear',
                              align_corners=True),
                F.interpolate(d4, scale_factor=8, mode='bilinear',
                              align_corners=True),
                F.interpolate(d5, scale_factor=16, mode='bilinear',
                              align_corners=True)
            ), 1
        )  # [B, 320, H, W]
        logit = self.logit(f)  # [B, num_classes, H, W]
        return logit
