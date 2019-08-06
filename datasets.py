import os

from PIL import Image
import torch
import torchvision


class SegmentationDataset(torch.utils.data.Dataset):
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

    def __init__(self, X,  y, num_classes, transform=None):
        self.num_classes = num_classes
        self.X = X
        self.y = y
        self._check_images_exist()

        if transform is None:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ])
        self.transform = transform

    def __len__(self):
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
        y = Image.open(self.y[idx]).convert('L')

        x = self.transform(x)
        y = self.transform(y) * 255

        return x, y.long()