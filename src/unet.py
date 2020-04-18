import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 512)

        self.up1 = UpBlock(1024, 256)
        self.up2 = UpBlock(512, 128)
        self.up3 = UpBlock(256, 64)
        self.up4 = UpBlock(128, 64)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.out_conv(x)
        return torch.sigmoid(x)


if __name__ == '__main__':
    model = UNet(3, 10)

    x = torch.zeros((2, 3, 128, 128))
    print(model(x).size())