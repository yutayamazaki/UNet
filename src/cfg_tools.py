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
