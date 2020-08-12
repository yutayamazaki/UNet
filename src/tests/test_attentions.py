import unittest

import torch
import torch.nn as nn

from models import attentions


class sSETests(unittest.TestCase):

    def test_return_shape(self):
        atten: nn.Module = attentions.sSE(4)
        size: torch.Size = torch.Size((2, 4, 10, 10))
        inputs: torch.Tensor = torch.zeros(size)
        out = atten(inputs)
        self.assertEqual(out.size(), size)


class cSETests(unittest.TestCase):

    def test_return_shape(self):
        atten: nn.Module = attentions.cSE(4)
        size: torch.Size = torch.Size((2, 4, 10, 10))
        inputs: torch.Tensor = torch.zeros(size)
        out = atten(inputs)
        self.assertEqual(out.size(), size)


class scSETests(unittest.TestCase):

    def test_return_shape(self):
        atten: nn.Module = attentions.scSE(4)
        size: torch.Size = torch.Size((2, 4, 10, 10))
        inputs: torch.Tensor = torch.zeros(size)
        out = atten(inputs)
        self.assertEqual(out.size(), size)


class FPATests(unittest.TestCase):

    def test_return_shape(self):
        atten: nn.Module = attentions.FPA(
            in_channels=8, out_channels=4, fmap_size=16
        )
        size: torch.Size = torch.Size((2, 8, 16, 16))
        inputs: torch.Tensor = torch.zeros(size)
        out = atten(inputs)
        self.assertEqual(out.size(), torch.Size((2, 4, 16, 16)))
