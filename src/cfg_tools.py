import torch


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
