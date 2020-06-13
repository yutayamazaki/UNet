import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """ Focal Loss: https://arxiv.org/abs/1708.02002

    Args:
        gamma (float): It modulates the loss for each classes.
        size_average (bool): If True, it returns mean, else it returns sum.
    """
    def __init__(self, gamma: float = 2.0, size_average: bool = True) -> None:
        super(FocalLoss, self).__init__()
        self.gamma: float = gamma
        self.size_average: bool = size_average
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(  # type: ignore
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            targets (torch.Tensor) : Truth label with shape (B, ).
            outputs (torch.Tensor): Prediction result with shape
                (B, num_classes).
        Returns:
            torch.Tensor: Calculated FocalLoss.
        """
        logpt = self.cross_entropy(outputs, targets)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt  # type: ignore
        if self.size_average:
            return loss.mean()
        return loss.sum()
