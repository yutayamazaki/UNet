
import torch


def hflip(x: torch.Tensor) -> torch.Tensor:
    """Apply horizontal flip to torch.Tensor.
    Args:
        x (torch.Tensor): Torch tensor with shape (B, C, H, W).
    Returns:
        torch.Tensor: Horizontally fliped image.
    """
    return torch.flip(x, dims=[3])


def vflip(x: torch.Tensor) -> torch.Tensor:
    """Apply vertical flip to torch.Tensor.
    Args:
        x (torch.Tensor): Torch tensor with shape (B, C, H, W).
    Returns:
        torch.Tensor: Vertically fliped image.
    """
    return torch.flip(x, dims=[2])
