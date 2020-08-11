from typing import List

import torch
import torch.nn as nn

from tta import functional as F


class BaseTransform:

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class EmptyTransform(BaseTransform):

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class HorizontalFlip(BaseTransform):

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.hflip(x)


class VerticalFlip(BaseTransform):

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.vflip(x)


class TTAWrapper(nn.Module):

    def __init__(
        self, net: nn.Module, transforms: List[BaseTransform],
        mode: str = 'max'
    ):
        super(TTAWrapper, self).__init__()
        self.net: nn.Module = net
        self.transforms: List[BaseTransform] = transforms
        self.mode: str = mode

    @staticmethod
    def _max(x: List[torch.Tensor]) -> torch.Tensor:
        ret, _ = torch.stack(x, dim=0).max(dim=0)
        return ret

    @staticmethod
    def _mean(x: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(x, dim=0).mean(dim=0)

    def forward(self, x):
        outputs: List[torch.Tensor] = []
        for transform in self.transforms:
            x_trans: torch.Tensor = transform(x)
            out: torch.Tensor = self.net(x_trans)
            outputs.append(transform(out))

        if self.mode == 'max':
            return self._max(outputs)
        elif self.mode == 'mean':
            return self._mean(outputs)
        else:
            raise ValueError('TTWrapper.mode must be "max", "mean".')
