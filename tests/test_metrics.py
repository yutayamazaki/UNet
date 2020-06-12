import unittest
from typing import List

import numpy as np
import torch

from src import metrics


class MeanIntersectionOverUnionTests(unittest.TestCase):

    def setUp(self):
        self.num_classes: int = 2
        self.y_true: torch.Tensor = torch.zeros((2, 4, 4))
        self.y_pred: torch.Tensor = torch.zeros((2, self.num_classes, 4, 4))

    def test_mean_iou_zero(self):
        mean_iou: float = metrics.mean_intersection_over_union(
            self.y_true, self.y_pred, self.num_classes
        )
        self.assertIsInstance(mean_iou, float)
        self.assertEqual(mean_iou, 0.0)


class IntersectionOverUnionTests(unittest.TestCase):

    def setUp(self):
        self.num_classes: int = 2
        self.y_true: torch.Tensor = torch.zeros((2, 4, 4))
        self.y_pred: torch.Tensor = torch.zeros((2, self.num_classes, 4, 4))

    def test_return_type(self):
        iou_list: List[float] = metrics.intersection_over_union(
            self.y_true, self.y_pred, self.num_classes
        )
        self.assertIsInstance(iou_list, list)
        self.assertEqual(tuple(iou_list), (0., 0.))
