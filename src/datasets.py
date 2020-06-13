import os

import torchvision
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SegmentationDataset(Dataset):
    """ PyTorch Dataset for Semnatic Segmentation.
    Parameters
    ----------
    X: list of str
        Paths to train image.

    y: list of str
        Paths to ground truth image.

    transform: torchvision.transforms
        Transform for image.
    """
    def __init__(self, X,  y, num_classes: int,
                 transform=None, img_size: int = 256):
        self.num_classes: int = num_classes
        self.X = X
        self.y = y
        self.img_size: int = img_size
        self._check_images_exist()

        if transform is None:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.img_size, self.img_size)),
                torchvision.transforms.ToTensor(),
            ])
        self.transform = transform

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
        x = Image.open(self.X[idx]).convert('RGB')
        y = Image.open(self.y[idx])

        x = self.transform(x)
        y = self.transform(y) * 255
        y[y == 255] = 21  # Background
        return x, y.long()
