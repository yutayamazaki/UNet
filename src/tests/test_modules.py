import unittest
from typing import Tuple

import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from models import modules


class LoadResNetBackboneTests(unittest.TestCase):

    def setUp(self):
        self.backbones: Tuple[str] = (
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        )

    def test_return(self):
        pretrained: bool = False
        for backbone in self.backbones:
            net = modules.load_resnet_backbone(backbone, pretrained)

            has_basic_block: bool = backbone in ('resnet18', 'resnet34')
            block = BasicBlock if has_basic_block else Bottleneck
            self.assertIsInstance(net.layer1[0], block)

    def test_raise(self):
        with self.assertRaises(ValueError):
            pretrained: bool = False
            modules.load_resnet_backbone('invalid_backbone', pretrained)


class Conv3x3Tests(unittest.TestCase):

    def test_return(self):
        conv: nn.Module = modules.conv3x3(2, 3)
        self.assertEqual(len(conv), 3)  # conv has (conv2d, bn, relu)
