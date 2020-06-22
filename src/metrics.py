from typing import List

import numpy as np
import torch

from losses import dice_loss


def intersection_and_union(
    y_true: torch.Tensor, y_pred: torch.Tensor,
    num_classes: int, ignore_index: int = 255
):
    """Calculate intersection and union for given labels and predictions.
    Args:
        y_true (torch.Tensor): Ground truth.
        y_pred (torch.Tensor): Predictions.
        num_classes (int): A unmber of unique classes.
        ignore_index (int): Specify value which ignore like background.

    Returns:
        tuple: A tuple of intersection and union.
    """
    assert ignore_index > num_classes, \
        'ignore_index should be grater than num_classes.'
    assert y_true.size() == y_pred.size(), \
        'Shape of y_true and y_pred must be same.'

    y_pred = y_pred.byte().flatten()
    y_true = y_true.byte().flatten()

    # Ignore specified index.
    is_not_ignore = y_true != ignore_index
    y_pred = y_pred[is_not_ignore]
    y_true = y_true[is_not_ignore]

    intersection = y_pred[y_pred == y_true]
    area_intersection = intersection.bincount(minlength=num_classes)

    bincount_pred = y_pred.bincount(minlength=num_classes)
    bincount_true = y_true.bincount(minlength=num_classes)

    area_union = bincount_pred + bincount_true - area_intersection
    area_intersection = area_intersection.float().cpu().numpy()
    area_union = area_union.float().cpu().numpy()

    return area_intersection, area_union


def mean_intersection_over_union(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int
) -> float:
    """Return IoU metric average over classes.
    Args:
        y_true (torch.Tensor): With shape (B, H, W).
        y_pred (torch.Tensor): With shape (B, num_classes, H, W).
        num_classes (int): A number of  unique classes.
    Returns:
        float: Mean IoU.
    """
    y_pred = y_pred.argmax(dim=1)
    intersection, union = intersection_and_union(
        y_true=y_true, y_pred=y_pred, num_classes=num_classes
    )
    mean_iou: np.foat32 = np.mean(intersection / (union + 1e-16))
    return float(mean_iou)


def intersection_over_union(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int
) -> List[float]:
    """Return IoU metric of each classes.
    Args:
        y_true (torch.Tensor): With shape (B, H, W).
        y_pred (torch.Tensor): With shape (B, num_classes, H, W).
        num_classes (int): A number of  unique classes.
    Returns:
        list: A list of float for each classes.
    """
    y_pred = y_pred.argmax(dim=1)
    intersection, union = intersection_and_union(
        y_true=y_true, y_pred=y_pred, num_classes=num_classes
    )
    return [float(iou) for iou in intersection / (union + 1e-16)]


def dice_coefficient(
    inputs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8
) -> float:
    return 1. - float(dice_loss(inputs, targets, eps))
