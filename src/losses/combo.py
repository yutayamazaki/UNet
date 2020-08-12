from typing import Optional, Tuple

import torch.nn as nn

from .jaccard import JaccardLoss


class ComboLoss(nn.Module):
    """ComboLoss is a weighted loss of JaccardLoss and CrossEntropyLoss.
    Args:
        num_classes (int): A number of unique classes.
        ignore_index (Optional[int]): Specify index to ignore classes.
        weights (Tuple[float, float]): Weights for JaccardLoss and
                                       CrossEntropyLoss.
    """
    def __init__(
        self, num_classes: int, ignore_index: Optional[int] = None,
        weights: Tuple[float, float] = (0.5, 0.5)
    ):
        super(ComboLoss, self).__init__()
        self.jaccard_loss: JaccardLoss = JaccardLoss(num_classes, ignore_index)
        self.cross_entropy: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        assert len(weights) == 2, 'len(weights) must be 2.'
        self.weights: Tuple[float, float] = weights

    def forward(  # type: ignore
        self,
        inputs,
        targets
    ):
        """
        Args:
            inputs (torch.Tensor): (B, num_classes, H, W)
            targets (torch.Tensor): (B, H, W)
        Returns:
            torch.Tensor: Computed ComboLoss.
        """
        jaccard = self.jaccard_loss(inputs, targets)
        ce = self.cross_entropy(inputs, targets)
        return self.weights[0] * jaccard + self.weights[1] * ce
