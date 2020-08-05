import unittest

import torch
from torchvision.models.resnet import BasicBlock, Bottleneck

import models


class LoadModelTests(unittest.TestCase):

    def test_unet(self):
        model = models.load_unet(backbone=None, num_classes=2)
        self.assertIsInstance(model, models.UNet)

    def test_unet_resnet(self):
        model = models.load_unet(backbone='resnet34', num_classes=2)
        self.assertIsInstance(model, models.UNetResNet)

    def test_raise_name(self):
        with self.assertRaises(ValueError):
            models.load_unet(backbone='InvalidName', num_classes=2)


class LoadResNetBackboneTests(unittest.TestCase):

    def setUp(self):
        self.backbones = (
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2',
            'wide_resnet101_2'
        )

    def test_return(self):
        pretrained: bool = False
        for backbone in self.backbones:
            net = models.modules.load_resnet_backbone(
                backbone, pretrained
            )

            has_basic_block: bool = backbone in ('resnet18', 'resnet34')
            block = BasicBlock if has_basic_block else Bottleneck
            self.assertIsInstance(net.layer1[0], block)

    def test_raise(self):
        with self.assertRaises(ValueError):
            pretrained: bool = False
            models.modules.load_resnet_backbone(
                'invalid_backbone', pretrained
            )


class UNetResNetTests(unittest.TestCase):

    def setUp(self):
        self.backbones = (
            'resnet18',  # BasicBlock
            'resnet50',  # BottleNeck
            'resnext50_32x4d',  # ResNeXt, BottleNeck
            'wide_resnet50_2'  # WIdeResNet, BottleNeck
        )

    def test_output_shape(self):
        pretrained: bool = False
        num_classes: int = 3
        for backbone in self.backbones:
            net = models.UNetResNet(num_classes, backbone, pretrained)

            size: torch.Size = torch.Size([2, num_classes, 256, 256])
            x: torch.Tensor = torch.zeros(size)
            out: torch.Tensor = net(x)
            self.assertEqual(out.size(), size)
