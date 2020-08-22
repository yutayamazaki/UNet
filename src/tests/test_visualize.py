import unittest
from typing import List, Tuple

import numpy as np

import visualize as vis


class ApplyMaskOnImageTests(unittest.TestCase):

    def setUp(self):
        self.img: np.ndarray = np.zeros((2, 10, 10))
        self.img[0, 0, 0] = 1
        self.cmaps: List[Tuple[str, Tuple[int]]] = [
            ('class_name_1', (0, 128, 128)),
            ('class_name_2', (0, 0, 128))
        ]

    def test_return_simple(self):
        out = vis.apply_mask_on_image(
            mask=self.img,
            image=np.zeros((10, 10, 3)),
            cmaps=self.cmaps
        )
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (10, 10, 3))
        self.assertEqual(tuple(out[0][0]), self.cmaps[0][1])


class VisualizeErrorsTests(unittest.TestCase):

    def setUp(self):
        self.pred_mask: np.ndarray = np.zeros((2, 10, 10))
        self.gt_mask = np.zeros((10, 10))
        self.cmaps: List[Tuple[str, Tuple[int]]] = [
            ('class_name_1', (0, 128, 128)),
            ('class_name_2', (0, 0, 128))
        ]

    def test_return_simple(self):
        out1 = vis.visualize_errors(
            pred_mask=self.pred_mask,
            gt_mask=self.gt_mask,
            image=np.zeros((10, 10, 3)),
            cmaps=self.cmaps,
            error_type='fn'
        )
        out2 = vis.visualize_errors(
            pred_mask=self.pred_mask,
            gt_mask=self.gt_mask,
            image=np.zeros((10, 10, 3)),
            cmaps=self.cmaps,
            error_type='fp'
        )
        self.assertEqual(out1.shape, (10, 10, 3))
        self.assertEqual(out2.shape, (10, 10, 3))

    def test_raise(self):
        with self.assertRaises(ValueError):
            vis.visualize_errors(
                pred_mask=self.pred_mask,
                gt_mask=self.gt_mask,
                image=np.zeros((10, 10, 3)),
                cmaps=self.cmaps,
                error_type='raise-error'
            )
