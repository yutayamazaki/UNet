import torch
import torch.nn as nn


class sSE(nn.Module):

    """ Implementation of 'Channel Squeeze and spatial excitation; sSE'.
    Note:
        https://arxiv.org/abs/1709.01507
    """

    def __init__(self, in_channel):
        super(sSE, self).__init__()
        self.squeeze = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)

    def forward(self, x):
        """ [B, C, H, W] -> [B, C, H, W] """
        sq = self.squeeze(x)
        return x * torch.sigmoid(sq)


class cSE(nn.Module):

    """ Implementation of 'Spatial squeeze and Channel Excitation; cSE'.
    Note:
        https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels, reduction=4):
        super(cSE, self).__init__()
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


class scSE(nn.Module):

    """ Implementation of scSE module.
    Note:
        https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels):
        super(scSE, self).__init__()
        self.spatial_att = sSE(in_channels)
        self.channel_att = cSE(in_channels)

    def forward(self, x):
        """ [B, C, H, W] -> [B, C, H, W] """
        return self.spatial_att(x) + self.channel_att(x)
