import unittest
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

import utils


class ResizeMaskTests(unittest.TestCase):

    def test_return(self):
        mask: torch.Tensor = torch.randn((5, 10, 20))
        height: int = 5
        width: int = 15

        resized_mask: torch.Tensor = utils.resize_mask(mask, height, width)
        self.assertEqual(resized_mask.size(), torch.Size((5, height, width)))


class ResizeMasksTests(unittest.TestCase):

    def test_return(self):
        masks: torch.Tensor = torch.randn((2, 5, 10, 20))
        height: int = 15
        width: int = 25

        resized_masks: torch.Tensor = utils.resize_masks(
            masks, height, width
        )
        self.assertEqual(
            resized_masks.size(), torch.Size((2, 5, height, width))
        )


class ResizeTensorImageTests(unittest.TestCase):

    def test_simple(self):
        x: torch.Tensor = torch.randn((3, 10, 10))
        height: int = 5
        width: int = 8
        resized: torch.Tensor = utils.resize_tensor_image(x, height, width)

        self.assertEqual(resized.size(), torch.Size((3, height, width)))


class ResizeTensorImagesTests(unittest.TestCase):

    def test_simple(self):
        x: torch.Tensor = torch.randn((2, 3, 10, 10))
        height: int = 5
        width: int = 8
        resized: torch.Tensor = utils.resize_tensor_images(x, height, width)

        self.assertEqual(resized.size(), torch.Size((2, 3, height, width)))


def _is_same_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.all(torch.eq(a, b))


class ToOneHotTests(unittest.TestCase):

    def test_simple(self):
        mask: torch.Tensor = torch.randint(low=0, high=2, size=(2, 15, 10))
        num_classes: int = 3
        oh_mask: torch.Tensor = utils.to_one_hot(mask, num_classes)

        self.assertEqual(oh_mask.size(), torch.Size((2, num_classes, 15, 10)))
        self.assertTrue(_is_same_tensor(oh_mask.argmax(dim=1), mask))


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
