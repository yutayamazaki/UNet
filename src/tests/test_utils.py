import unittest
from typing import Any, Dict, List, Tuple

import numpy as np
import torch.nn as nn

import utils


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


class CountParametersTests(unittest.TestCase):

    def test_simple(self):
        net: nn.Module = nn.Linear(3, 2, bias=False)
        num_params: int = utils.count_parameters(net)
        self.assertEqual(num_params, 3 * 2)


class DotDictTests(unittest.TestCase):

    def test_simple(self):
        dic: Dict[str, Any] = {
            'string': 'aaa',
            'integer': 12,
            'dict': {'key': 'value'},
            'list': [1, 2]
        }
        dotdict = utils.DotDict(dic)
        self.assertEqual(dotdict.string, dic['string'])
        self.assertEqual(dotdict.integer, dic['integer'])
        self.assertEqual(dotdict.dict.key, dic['dict']['key'])
        self.assertEqual(dotdict.list, dic['list'])

    def test_todict(self):
        dic: Dict[str, Any] = {
            'nested': {'key': {'key': 'val'}}
        }
        dotdict = utils.DotDict(dic)
        ret: Dict[Any, Any] = dotdict.todict()

        self.assertDictEqual(ret, dic)
