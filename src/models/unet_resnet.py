from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import attentions, modules


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


class Decoder(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = modules.conv3x3(in_channels, channels)
        self.conv2 = modules.conv3x3(channels, out_channels)
        self.s_att = attentions.sSE(out_channels)
        self.c_att = attentions.cSE(out_channels, 16)

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
        self.s_att = attentions.sSE(n_out)
        self.c_att = attentions.cSE(n_out, 16)

    def forward(self, x, e):
        """ [B, C, H, W] -> [B, C, H*2, W*2] """
        x = self.tr_conv(x)
        e = self.x_conv(e)

        cat = torch.cat([x, e], 1)
        cat = self.relu(self.bn(cat))
        s = self.s_att(cat)
        c = self.c_att(cat)
        return s + c


class UNetResNet(nn.Module):

    def __init__(
        self, num_classes: int, backbone: str = 'resnet18',
        pretrained: bool = True
    ):
        super(UNetResNet, self).__init__()
        resnet: nn.Module = modules.load_resnet_backbone(backbone, pretrained)

        num_channels: List[int] = self._params_bottleneck_block()
        if backbone in ('resnet18', 'resnet34'):
            num_channels = self._params_basic_block()

        self.backbone = backbone

        self.conv1 = nn.Sequential(
            resnet.conv1,  # type: ignore
            resnet.bn1,  # type: ignore
            resnet.relu  # type: ignore
        )

        self.encode2 = nn.Sequential(
            resnet.layer1,  # type: ignore
            attentions.scSE(in_channels=num_channels[0])
        )
        self.encode3 = nn.Sequential(
            resnet.layer2,  # type: ignore
            attentions.scSE(in_channels=num_channels[1])
        )
        self.encode4 = nn.Sequential(
            resnet.layer3,  # type: ignore
            attentions.scSE(in_channels=num_channels[2])
        )
        self.encode5 = nn.Sequential(
            resnet.layer4,  # type: ignore
            attentions.scSE(in_channels=num_channels[3])
        )

        self.center = nn.Sequential(
            FPA(in_channels=num_channels[3], out_channels=num_channels[2]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decode5 = Decoderv2(num_channels[2], num_channels[3], 64)
        self.decode4 = Decoderv2(64, num_channels[2], 64)
        self.decode3 = Decoderv2(64, num_channels[1], 64)
        self.decode2 = Decoderv2(64, num_channels[0], 64)
        self.decode1 = Decoder(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, num_classes, kernel_size=1, bias=False)
        )

    @staticmethod
    def _params_basic_block() -> List[int]:
        """Return numbers of feature maps used in resnet18 and resnet34."""
        return [64, 128, 256, 512]

    @staticmethod
    def _params_bottleneck_block() -> List[int]:
        """Return numbers of feature maps used in resnet50, 101, 152."""
        return [256, 512, 1024, 2048]

    def _get_model_name(self) -> str:
        """Return numbers of feature maps used in resnet50, 101, 152."""
        return f'unet_{self.backbone}'

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
