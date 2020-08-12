import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Interpolate(nn.Module):
    """Wrapper class of torch.nn.functional.interpolate."""
    def __init__(
        self, scale_factor: int, mode: str, align_corners: bool = True
    ):
        super(Interpolate, self).__init__()
        self.scale_factor: int = scale_factor
        self.mode: str = mode
        self.align_corners: bool = align_corners

    def forward(self, x):
        return F.interpolate(
            x, scale_factor=self.scale_factor,
            mode=self.mode, align_corners=self.align_corners
        )


class FPA(nn.Module):
    """Implementation of Feature Pyramid Attention.
       https://arxiv.org/abs/1805.10180

    Args:
        in_channels (int): Number of input channels.
        out_chanels (int): Number of output channels.
        fmap_size (int): Height and width size of input feature maps.

    Examples:
        >>> net = FPA(512, 256, 16)
        >>> x: torch.Tensor = torch.randn((2, 512, 16, 16))
        >>> print(net(x).size())  # torch.Size([2, 256, 16, 16])
    """
    def __init__(self, in_channels: int, out_channels: int, fmap_size: int):
        super(FPA, self).__init__()
        self.fmap_size: int = fmap_size
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            Interpolate(
                scale_factor=fmap_size, mode='bilinear', align_corners=True
            )
        )
        self.conv7x7_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv7x7_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv5x5_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=2,
                      padding=2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )
        self.conv5x5_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, in_channels, H, W).

        Returns:
            torch.Tensor: With shape (B, out_channels, H, W)

        Warnings:
            Height and width of given feature maps must be 16.
        """
        assert x.size()[2] == self.fmap_size and x.size()[3] == self.fmap_size
        x_gap = self.gap(x)  # (B, out_channels, H, W)

        e1 = self.conv7x7_1(x)  # (B, in_channels, H/2, W/2)
        atten7x7 = self.conv7x7_2(e1)  # (B, out_channels, H/2, W/2)
        e2 = self.conv5x5_1(e1)  # (B, in_channels, H/4, W/4)
        atten5x5 = self.conv5x5_2(e2)  # (B, out_channels, H/4, W/4)
        e3 = self.conv3x3_1(e2)  # (B, in_channels, H/8, W/8)
        atten3x3 = self.conv3x3_2(e3)  # (B, out_channels, H/8, W/8)

        atten3x3 = F.interpolate(
            atten3x3, scale_factor=2, mode='bilinear', align_corners=True
        )  # (B, out_channels, H/4, W/4)
        atten5x5 = atten3x3 + atten5x5

        atten5x5 = F.interpolate(
            atten5x5, scale_factor=2, mode='bilinear', align_corners=True
        )  # (B, out_channels, H/2, W/2)
        atten7x7 = atten5x5 + atten7x7

        attention = F.interpolate(
            atten7x7, scale_factor=2, mode='bilinear', align_corners=True
        )  # (B, out_channels, H, W)

        x = self.conv1x1(x)  # (B, out_channels, H, W)
        x = x * attention
        return x + x_gap
