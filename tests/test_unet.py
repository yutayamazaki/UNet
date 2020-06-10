import unittest
from typing import List, Tuple

import numpy as np

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
