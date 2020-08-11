import unittest

import torch

import tta.functional as F
import tta.transforms as transforms


class EmptyTransformTests(unittest.TestCase):

    def test_same_as_functional(self):
        x: torch.Tensor = torch.randn((1, 3, 24, 24))
        transform = transforms.EmptyTransform()
        x_inv: torch.Tensor = transform(x)

        self.assertTrue(torch.equal(x_inv, x))


class HorizontalFlipTests(unittest.TestCase):

    def test_same_as_functional(self):
        x: torch.Tensor = torch.randn((1, 3, 24, 24))
        transform = transforms.HorizontalFlip()
        x_inv: torch.Tensor = transform(x)

        self.assertTrue(torch.equal(x_inv, F.hflip(x)))


class VerticalFlipTests(unittest.TestCase):

    def test_same_as_functional(self):
        x: torch.Tensor = torch.randn((1, 3, 24, 24))
        transform = transforms.VerticalFlip()
        x_inv: torch.Tensor = transform(x)

        self.assertTrue(torch.equal(x_inv, F.vflip(x)))


class TTAWrapperTests(unittest.TestCase):

    def setUp(self):
        self.net = torch.relu
        self.transforms = [transforms.HorizontalFlip()]
        self.mode: str = 'max'

    def test_mode_max(self):
        size: torch.Size = torch.Size((1, 3, 5, 5))
        x: torch.Tensor = torch.randn(size)
        wrapper = transforms.TTAWrapper(self.net, self.transforms, self.mode)
        out = wrapper(x)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.size(), size)

    def test_mode_mean(self):
        size: torch.Size = torch.Size((1, 3, 5, 5))
        x: torch.Tensor = torch.randn(size)
        wrapper = transforms.TTAWrapper(self.net, self.transforms, 'mean')
        out = wrapper(x)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.size(), size)
