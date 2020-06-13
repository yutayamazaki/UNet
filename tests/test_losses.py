import unittest

import torch

from src import losses


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
