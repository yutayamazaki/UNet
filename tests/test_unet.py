import unittest
from typing import List, Tuple

import torch
from torchvision.models.resnet import BasicBlock, Bottleneck

from src import unet


class LoadModelTests(unittest.TestCase):

    def test_unet(self):
        model = unet.load_model(name='UNet', num_classes=2)
        self.assertIsInstance(model, unet.UNet)

    def test_unet_resnet(self):
        model = unet.load_model(name='UNetResNet34', num_classes=2)
        self.assertIsInstance(model, unet.UNetResNet34)

    def test_raise_name(self):
        with self.assertRaises(ValueError):
            unet.load_model(name='InvalidName', num_classes=2)


class UNetResNet34Tests(unittest.TestCase):

    def test_output_shape(self):
        net = unet.UNetResNet34(num_classes=3)

        size: torch.Size = torch.Size((2, 3, 256, 256))
        x = torch.Tensor(size)
        out = net(x)
        self.assertEqual(out.size(), size)


class LoadResNetBackboneTests(unittest.TestCase):

    def setUp(self):
        self.backbones: Tuple[str] = (
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        )

    def test_return(self):
        pretrained: bool = False
        for backbone in self.backbones:
            net = unet._load_resnet_backbone(backbone, pretrained)

            has_basic_block: bool = backbone in ('resnet18', 'resnet34')
            block = BasicBlock if has_basic_block else Bottleneck
            self.assertIsInstance(net.layer1[0], block)

    def test_raise(self):
        with self.assertRaises(ValueError):
            pretrained: bool = False
            unet._load_resnet_backbone('invalid_backbone', pretrained)


class UNetResNetTests(unittest.TestCase):

    def setUp(self):
        self.backbones: Tuple[str] = (
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        )

    def test_output_shape(self):
        pretrained: bool = False
        num_classes: int = 3
        for backbone in self.backbones:
            net = unet.UNetResNet(num_classes, backbone, pretrained)

            size: torch.Size = torch.Size([2, num_classes, 256, 256])
            x: torch.Tensor = torch.zeros(size)
            out: torch.Tensor = net(x)
            self.assertEqual(out.size(), size)
