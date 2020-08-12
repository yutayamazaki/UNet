import os
from typing import Callable, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """ PyTorch Dataset for Semnatic Segmentation.
    Args:
        X (List[str]): Paths to each images.
        y (List[str]): Paths to each mask images.
        num_classes (int): A number of unique classes.
        img_size (int): Height and width.
        transforms (Callable): Callabel instance of albumentations.
    """
    def __init__(
        self, X: List[str], y: List[str], num_classes: int,
        transforms: Callable, img_size: int = 256
    ):
        self.num_classes: int = num_classes
        self.X = X
        self.y = y
        self.img_size: int = img_size
        self._check_images_exist()
        self.transforms: Callable = transforms

    def __len__(self) -> int:
        return len(self.X)

    def _check_images_exist(self):
        X_, y_ = [], []
        for x, y in zip(self.X, self.y):
            if os.path.exists(x) and os.path.exists(x):
                X_.append(x)
                y_.append(y)
            else:
                print(f'Not found {x} or {y}.')
        self.X = X_
        self.y = y_

    def __getitem__(self, idx):
        """
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                img.size() == torch.Size([3, H, W])
                mask.size() == torch.Size([1, H, W])
        """
        pil_img: Image.Image = Image.open(self.X[idx]).convert('RGB')
        pil_mask: Image.Image = Image.open(self.y[idx])

        img_arr: np.ndarray = np.array(pil_img)
        mask_arr: np.ndarray = np.array(pil_mask)
        aug: Dict[str, np.ndarray] = self.transforms(
            image=img_arr, mask=mask_arr
        )
        img: torch.Tensor = torch.as_tensor(
            aug['image'].transpose(2, 0, 1)
        ).float()
        mask: torch.Tensor = torch.as_tensor(
            (aug['mask'])
        ).unsqueeze(0)

        mask[mask == 255] = 21  # Background
        return img, mask.long()
