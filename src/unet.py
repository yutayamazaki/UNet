import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def conv_relu(in_channels, out_channels):
    return nn.Sequential(
               nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(inplace=True)
            )


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
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

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True)
        self.conv1 = conv_relu(in_channels, out_channels)
        self.conv2 = conv_relu(out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):

    def __init__(self, in_channels, num_classes):
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


class FPA(nn.Module):

    def __init__(self, in_channels, out_channels):
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


def conv3x3(input_dim, output_dim, rate=1):
    conv = nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=3, dilation=rate,
                  padding=rate, bias=False),
        nn.BatchNorm2d(output_dim),
        nn.ReLU(True)
    )
    return conv


class Decoder(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = conv3x3(in_channels, channels)
        self.conv2 = conv3x3(channels, out_channels)
        self.s_att = sSE(out_channels)
        self.c_att = cSE(out_channels, 16)

    def forward(self, x, e=None):
        """ [B, C, H, W] -> [B, C, H*2, W*2] """
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=True
        )
        if e is not None:
            print(f'{x.shape}, {e.shape}')
            x = torch.cat([x, e], 1)
            print(f'{x.shape, {e.shape}}')
        x = self.conv1(x)
        x = self.conv2(x)
        s = self.s_att(x)
        c = self.c_att(x)
        output = s + c
        return output


class Decoderv2(nn.Module):

    def __init__(self, up_in, x_in, n_out):
        super(Decoderv2, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1, bias=False)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(True)
        self.s_att = sSE(n_out)
        self.c_att = cSE(n_out, 16)

    def forward(self, x, e):
        """ [B, C, H, W] -> [B, C, H*2, W*2] """
        x = self.tr_conv(x)
        e = self.x_conv(e)

        cat = torch.cat([x, e], 1)
        cat = self.relu(self.bn(cat))
        s = self.s_att(cat)
        c = self.c_att(cat)
        return s + c


class UNetResNet34(nn.Module):

    def __init__(self, num_classes: int):
        super(UNetResNet34, self).__init__()
        resnet = torchvision.models.resnet34(True)

        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )

        self.encode2 = nn.Sequential(resnet.layer1, scSE(64))
        self.encode3 = nn.Sequential(resnet.layer2, scSE(128))
        self.encode4 = nn.Sequential(resnet.layer3, scSE(256))
        self.encode5 = nn.Sequential(resnet.layer4, scSE(512))

        self.center = nn.Sequential(
            FPA(512, 256),
            nn.MaxPool2d(2, 2)
        )

        self.decode5 = Decoderv2(256, 512, 64)
        self.decode4 = Decoderv2(64, 256, 64)
        self.decode3 = Decoderv2(64, 128, 64)
        self.decode2 = Decoderv2(64, 64, 64)
        self.decode1 = Decoder(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, num_classes, kernel_size=1, bias=False)
        )

    def forward(self, x):
        """ [B, C, H, W] -> [B, num_classes, H, W] """
        x = self.conv1(x)  # [B, 64, H//2, W//2]
        e2 = self.encode2(x)  # [B, 64, H/2, W/2]
        e3 = self.encode3(e2)  # [B, 128, H/4, W/4]
        e4 = self.encode4(e3)  # [B, 256, H/8, W/8]
        e5 = self.encode5(e4)  # [B, 512, H/16, W/16]

        f = self.center(e5)  # [B, 256, H/32, W/32]

        d5 = self.decode5(f, e5)  # [B, 64, H/16, W/16]
        d4 = self.decode4(d5, e4)  # [B, 64, H/8, W/8]
        d3 = self.decode3(d4, e3)  # [B, 64, H/4, W/4]
        d2 = self.decode2(d3, e2)  # [B, 64, H/2, W/2]
        d1 = self.decode1(d2)  # [B, 256, H, W]

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


def load_model(name: str, num_classes: int) -> nn.Module:
    """Load specified segmentation model from unet.py"""
    lower_name: str = name.lower()
    if lower_name == 'unet':
        return UNet(in_channels=3, num_classes=num_classes)
    elif lower_name == 'unetresnet34':
        return UNetResNet34(num_classes=num_classes)
    else:
        raise ValueError('Argument [name] must be "UNet" or "UNetResNet34".')


if __name__ == '__main__':
    x = torch.zeros((2, 3, 256, 256))

    model = UNet(3, 10)
    print(model(x).size())
    model = UNetResNet34(num_classes=10)
    print(model(x).size())
