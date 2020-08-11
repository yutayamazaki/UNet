import unittest

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck

import models


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
            net = models.unet_resnet.UNetResNet(
                num_classes, backbone, pretrained
            )

            size: torch.Size = torch.Size([2, num_classes, 256, 256])
            x: torch.Tensor = torch.zeros(size)
            out: torch.Tensor = net(x)
            self.assertEqual(out.size(), size)


class LoadModelTests(unittest.TestCase):

    def setUp(self):
        self.num_classes: int = 2
        self.architecture = 'unet'
        self.backbone: str = 'resnet18'
        self.pretrained = False

    def test_load_unet(self):
        net: nn.Module = models.utils.load_model(
            self.num_classes, self.architecture, self.backbone,
            self.pretrained
        )
        self.assertIsInstance(net, models.unet_resnet.UNetResNet)
        self.assertEqual(net.backbone, self.backbone)

    def test_load_fpn(self):
        architecture: str = 'fpn'
        backbone: str = 'resnet34'
        net: nn.Module = models.utils.load_model(
            self.num_classes, architecture, backbone,
            self.pretrained
        )
        self.assertIsInstance(net, models.fpn.FPN)
        self.assertEqual(net.backbone, backbone)

    def test_raise(self):
        architecture: str = 'invalid-name'
        with self.assertRaises(ValueError):
            models.utils.load_model(
                self.num_classes, architecture, self.backbone,
                self.pretrained
            )
