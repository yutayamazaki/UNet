from typing import Tuple

import torch
import torch.nn as nn

import losses


def load_optimizer(params, name: str, **kwargs) -> torch.optim.Optimizer:
    """Load PyTorch optimizer."""
    if name == 'SGD':
        return torch.optim.SGD(
            params,
            **kwargs
        )
    else:
        raise ValueError('name must be "SGD."')


def load_scheduler(
    optimizer: torch.optim.Optimizer, name: str, **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    if name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **kwargs
        )
    elif name == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            **kwargs
        )
    else:
        msg: str = \
            'name must be CosineAnnealingLR or CosineAnnealingWarmRestarts.'
        raise ValueError(msg)


def load_loss(name: str, **kwargs) -> nn.Module:
    if name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(**kwargs)
    elif name == 'FocalLoss':
        return losses.FocalLoss(**kwargs)
    elif name == 'ComboLoss':
        return losses.ComboLoss(**kwargs)
    elif name == 'JaccardLoss':
        return losses.JaccardLoss(**kwargs)
    else:
        attributes: Tuple[str, ...] = (
            'CrossEntropyLoss', 'FocalLoss', 'ComboLoss', 'JaccardLoss'
        )
        raise ValueError(f'name must be in {attributes}.')
