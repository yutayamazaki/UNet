from typing import Optional

import torch
import torch.nn as nn


class JaccardLoss(nn.Module):
    """JaccardLoss optimize mIoU score directly.
    Args:
        num_classes (int): A number of unique classes.
        ignore_index (Optional[int]): Class label to ignore calculating score.
        eps (float): Used to prevent zero division.
    """
    def __init__(
        self, num_classes: int, ignore_index: Optional[int] = None,
        eps: float = 1e-16
    ):
        super(JaccardLoss, self).__init__()
        self.num_classes: int = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(  # type: ignore
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.argmax(inputs, dim=1)
        inputs = inputs.byte().flatten()
        targets = targets.byte().flatten()

        if self.ignore_index is not None:
            is_not_ignore = targets != self.ignore_index
            inputs = inputs[is_not_ignore]
            targets = targets[is_not_ignore]

        intersection = inputs[inputs == targets]
        area_intersection = intersection.bincount(minlength=self.num_classes)

        bincount_pred = inputs.bincount(minlength=self.num_classes)
        bincount_true = targets.bincount(minlength=self.num_classes)

        area_union = bincount_pred + bincount_true - area_intersection

        mean_iou = torch.mean(area_intersection / (area_union + self.eps))
        return mean_iou
