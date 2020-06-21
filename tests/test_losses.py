import unittest

import torch

from src import losses
from src.losses.dice_loss import _to_one_hot


class FocalLossTests(unittest.TestCase):

    def setUp(self):
        self.outputs = torch.Tensor(([0.1, 0.9], [0.8, 0.2]))
        self.targets = torch.Tensor([1, 0]).long()
        self.criterion = losses.FocalLoss()

    def test_return_simple(self):
        loss = self.criterion(self.outputs, self.targets)
        self.assertIsInstance(loss, torch.Tensor)

    def test_check_size_average(self):
        criterion = losses.FocalLoss(size_average=True)
        loss_mean = criterion(self.outputs, self.targets)

        criterion = losses.FocalLoss(size_average=False)
        loss_sum = criterion(self.outputs, self.targets)

        self.assertEqual(loss_mean * len(self.targets), loss_sum)


class ToOneHotTests(unittest.TestCase):

    def test_simple(self):
        targets = torch.zeros((1, 5, 5))
        targets[0, 0, 0] = 1
        one_hot = _to_one_hot(targets, num_classes=2)
        self.assertEqual(one_hot.size(), torch.Size((1, 2, 5, 5)))
        self.assertEqual(one_hot[0, 0, 0, 0], torch.tensor(0))
        self.assertEqual(one_hot[0, 1, 0, 0], torch.tensor(1))


class DiceLossTests(unittest.TestCase):

    def setUp(self):
        self.outputs = torch.zeros((1, 2, 5, 5))
        self.outputs[0, 1, 0, 0] = 1
        self.targets = torch.zeros((1, 5, 5)).long()
        self.criterion = losses.DiceLoss()

    def test_return_simple(self):
        loss = self.criterion(self.outputs, self.targets)
        self.assertTrue(torch.isclose(loss, torch.tensor(0.5092423)))
