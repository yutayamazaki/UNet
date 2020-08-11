import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def load_labelmap(path: str) -> List[Tuple[str, Tuple[int]]]:
    """
    Args:
        path (str): A path to labelmap.txt generated by cvat.

    Returns:
        list: A list of tuple like [('class_name_1', (0, 128, 128)), (), ...].
    """
    color_maps: List[Tuple[str, Tuple[int]]] = []
    with open(path, 'r') as f:
        for line in f:
            if line[0] == '#':  # comment out
                continue
            class_name, cmap_str, _, _ = line.split(':')
            cmap: Tuple[int] = \
                tuple([int(c) for c in cmap_str.split(',')])  # type: ignore

            color_maps.append((class_name, cmap))
    return color_maps


def create_segmentation_result(
    img: np.ndarray, cmaps: List[Tuple[str, Tuple[int]]]
) -> np.ndarray:
    """ Apply colormaps to prediction result.
    Args:
        img (np.ndarray): Predicted image with shape
                          (NumClasses, Height, Width).
        cmaps (list): A list of tuple like
                      [('class_name_1': (0, 128, 128)), (), ...].
    Returns:
        np.ndarray: Visualized image using specified color maps with shape
                    (Height, Width, 3).
    """
    _, h, w = img.shape
    shape: Tuple[int] = (h, w, 3)  # type: ignore
    result_image: np.ndarray = np.zeros(shape)

    img_2d: np.ndarray = np.argmax(img, axis=0)
    for class_idx, (_, cmap) in enumerate(cmaps):
        # points: [[x, y], [x, y], ...]
        points: np.ndarray = np.argwhere(img_2d == class_idx)
        for x, y in points:
            result_image[x, y, :] = cmap

    return result_image


def count_parameters(net: nn.Module, requires_grad: bool = True) -> int:
    """Count the number of parameters given torch model."""
    if requires_grad:
        return sum(p.numel() for p in net.parameters() if p.requires_grad)
    return sum(p.numel() for p in net.parameters())


def seed_everything(seed: int = 1234):
    """Set seed for every modules."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore


class DotDict(dict):

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                self._parse_nested_dict(arg)

        if kwargs:
            self._parse_nested_dict(kwargs)

    def _parse_nested_dict(self, dic: Dict[Any, Any]):
        for k, v in dic.items():
            if isinstance(v, dict):
                self[k] = DotDict(v)
                continue
            self[k] = v

    def todict(self) -> Dict[Any, Any]:
        """Convert DotDict to default dict."""
        dic: Dict[Any, Any] = {}
        for k, v in self.items():
            if isinstance(v, DotDict):
                dic[k] = v.todict()
                continue
            dic[k] = v
        return dic

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]
