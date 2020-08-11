from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import modules


class FPNDecoder(nn.Module):

    def __init__(self, in_channels: int, cat_channels: int):
        super(FPNDecoder, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(
                in_channels=cat_channels, out_channels=in_channels,
                kernel_size=1, padding=0, stride=1,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(  # type: ignore
        self, inputs: torch.Tensor, concats: torch.Tensor
    ):
        # (B, C, H, W) -> (B, C, H/2m W/2)
        concats = self.conv1x1(concats)
        inputs = F.interpolate(
            inputs, scale_factor=2, mode='bilinear', align_corners=True
        )
        return concats + inputs


class FPNCenter(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(FPNCenter, self).__init__()
        self.conv = modules.conv3x3(in_channels, out_channels)

    def forward(self, inputs):
        # (B, in_channels, H, W) -> (B, out_channels, H, W)
        return self.conv(inputs)


class FPNSegmentationHead(nn.Module):

    def __init__(
        self, num_classes: int, in_channels: int = 256,
        out_channels: int = 128
    ):
        super(FPNSegmentationHead, self).__init__()
        self.conv1 = modules.conv3x3(in_channels, out_channels)
        self.conv2 = modules.conv3x3(in_channels, out_channels)
        self.conv3 = modules.conv3x3(in_channels, out_channels)
        self.conv4 = modules.conv3x3(in_channels, out_channels)
        self.conv3x3 = modules.conv3x3(out_channels * 4, num_classes)

    def forward(self, d1, d2, d3, d4):
        """
        Args:
            d1 (torch.Tensor): (B, in_channels, H/2, W/2)
            d2 (torch.Tensor): (B, in_channels, H/4, W/4)
            d3 (torch.Tensor): (B, in_channels, H/8, W/8)
            d4 (torch.Tensor): (B, in_channels, H/16, W/16)
        Returns:
            torch.Tensor: (B, num_classes, H, W)
        """
        d1 = F.interpolate(
            d1, scale_factor=2, mode='bilinear', align_corners=True
        )
        d2 = F.interpolate(
            d2, scale_factor=4, mode='bilinear', align_corners=True
        )
        d3 = F.interpolate(
            d3, scale_factor=8, mode='bilinear', align_corners=True
        )
        d4 = F.interpolate(
            d4, scale_factor=16, mode='bilinear', align_corners=True
        )
        out1 = self.conv1(d1)
        out2 = self.conv1(d2)
        out3 = self.conv1(d3)
        out4 = self.conv1(d4)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return self.conv3x3(out)


class FPN(nn.Module):
    """Feature Pyramid Networks.
    Args:
        num_classes (int): A number of classes used to specify output shape.
        backbone (str): ResNet backbone.
        pretrained (bool): Load pretrained resnet weights or not.
    Note:
        - https://arxiv.org/pdf/1612.03144.pdf
        - http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf
    """
    def __init__(
        self, num_classes: int, backbone: str = 'resnet18',
        pretrained: bool = True
    ):
        super(FPN, self).__init__()
        num_channels: List[int] = [256, 512, 1024, 2048]
        if backbone in ('resnet18', 'resnet34'):
            num_channels = [64, 128, 256, 512]

        self.backbone: str = backbone
        resnet = modules.load_resnet_backbone(backbone, pretrained)

        self.conv1 = nn.Sequential(
            resnet.conv1,  # type: ignore
            resnet.bn1,  # type: ignore
            resnet.relu  # type: ignore
        )

        self.encode1 = resnet.layer1
        self.encode2 = resnet.layer2
        self.encode3 = resnet.layer3
        self.encode4 = resnet.layer4

        self.center = FPNCenter(
            in_channels=num_channels[3], out_channels=num_channels[0]
        )

        self.decode1 = FPNDecoder(
            in_channels=num_channels[0], cat_channels=num_channels[0]
        )
        self.decode2 = FPNDecoder(
            in_channels=num_channels[0], cat_channels=num_channels[1]
        )
        self.decode3 = FPNDecoder(
            in_channels=num_channels[0], cat_channels=num_channels[2]
        )
        self.decode4 = FPNDecoder(
            in_channels=num_channels[0], cat_channels=num_channels[3]
        )

        self.seg_head = FPNSegmentationHead(
            num_classes=num_classes, in_channels=num_channels[0],
            out_channels=num_channels[0] // 2
        )

    def forward(self, inputs: torch.Tensor):  # type: ignore
        # (B, C, H, W) -> (B, C, H, W)
        inputs = self.conv1(inputs)  # (B, C, H/2, W/2)
        e1 = self.encode1(inputs)  # type: ignore # (B, C, H/4, W/4)
        e2 = self.encode2(e1)  # type: ignore # (B, C, H/8, W/8)
        e3 = self.encode3(e2)  # type: ignore # (B, C, H/16, W/16)
        e4 = self.encode4(e3)  # type: ignore # (B, C, H/32, W/32)
        d4 = self.center(e4)  # (B, C, H/16, W/16)
        d3 = self.decode3(inputs=d4, concats=e3)  # (B, C, H/8, W/8)
        d2 = self.decode2(inputs=d3, concats=e2)  # (B, C, H/4, W/4)
        d1 = self.decode1(inputs=d2, concats=e1)  # (B, C, H/2, W/2)
        return self.seg_head(d1, d2, d3, d4)  # (B, C, H, W)
