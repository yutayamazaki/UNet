import unittest

import torch

import tta.functional as F


class HFlipTests(unittest.TestCase):

    def test_simple(self):
        x: torch.Tensor = torch.randn((1, 3, 24, 24))
        x_inv: torch.Tensor = F.hflip(x)
        x_inv_inv: torch.Tensor = F.hflip(x_inv)

        self.assertTrue(torch.equal(x, x_inv_inv))
        self.assertEqual(x[0, 0, 0, 0], x_inv[0, 0, 0, -1])


class VFlipTests(unittest.TestCase):

    def test_simple(self):
        x: torch.Tensor = torch.randn((1, 3, 24, 24))
        x_inv: torch.Tensor = F.vflip(x)
        x_inv_inv: torch.Tensor = F.vflip(x_inv)

        self.assertTrue(torch.equal(x, x_inv_inv))
        self.assertEqual(x[0, 0, 0, 0], x_inv[0, 0, -1, 0])
