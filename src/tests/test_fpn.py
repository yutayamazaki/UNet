import unittest
from typing import Tuple

import torch

from models import fpn


class UNetResNetTests(unittest.TestCase):

    def setUp(self):
        self.backbones: Tuple[str] = (
            'resnet18', 'resnet34', 'resnet50',  # 'resnet101', 'resnet152'
        )

    def test_output_shape(self):
        pretrained: bool = False
        num_classes: int = 3
        for backbone in self.backbones:
            net = fpn.FPN(num_classes, backbone, pretrained)

            size: torch.Size = torch.Size([2, num_classes, 256, 256])
            x: torch.Tensor = torch.zeros(size)
            out: torch.Tensor = net(x)
            self.assertEqual(out.size(), size)
