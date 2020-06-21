from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def conv_relu(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
               nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(inplace=True)
            )


class DownBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = conv_relu(in_channels, out_channels)
        self.conv2 = conv_relu(out_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True
        )
        self.conv1 = conv_relu(in_channels, out_channels)
        self.conv2 = conv_relu(out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x//2,
                        diff_y // 2, diff_y - diff_y//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    """Original UNet implementation: https://arxiv.org/abs/1505.04597"""

    def __init__(self, in_channels: int, num_classes: int):
        super(UNet, self).__init__()
        self.in_conv = conv_relu(in_channels, 64)
        self.out_conv = conv_relu(64, num_classes)

        self.encode1 = DownBlock(64, 128)
        self.encode2 = DownBlock(128, 256)
        self.encode3 = DownBlock(256, 512)
        self.encode4 = DownBlock(512, 512)

        self.decode1 = UpBlock(1024, 256)
        self.decode2 = UpBlock(512, 128)
        self.decode3 = UpBlock(256, 64)
        self.decode4 = UpBlock(128, 64)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.encode1(x1)
        x3 = self.encode2(x2)
        x4 = self.encode3(x3)
        x5 = self.encode4(x4)

        x = self.decode1(x5, x4)
        x = self.decode2(x, x3)
        x = self.decode3(x, x2)
        x = self.decode4(x, x1)

        x = self.out_conv(x)
        return torch.sigmoid(x)
