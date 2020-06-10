import unittest
from typing import List, Tuple

import numpy as np

from src import utils


class CreateSegmentationResultTests(unittest.TestCase):

    def setUp(self):
        self.img: np.ndarray = np.zeros((2, 128, 128))
        self.img[0, 0, 0] = 1
        self.cmaps: List[Tuple[str, Tuple[int]]] = [
            ('class_name_1', (0, 128, 128)),
            ('class_name_2', (0, 0, 128))
        ]

    def test_return_simple(self):
        out = utils.create_segmentation_result(self.img, self.cmaps)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (128, 128, 3))
        self.assertEqual(tuple(out[0][0]), self.cmaps[0][1])
