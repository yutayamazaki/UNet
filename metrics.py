import numpy as np
import torch


def intersection_and_union(preds: torch.Tensor, labels: torch.Tensor,
                           ignore_index=255, num_classes=19):

    assert ignore_index > num_classes, 'ignore_index should be grater than n_classes'

    preds = preds.byte().flatten()
    labels = labels.byte().flatten()

    is_not_ignore = labels != ignore_index
    preds = preds[is_not_ignore]
    labels = labels[is_not_ignore]

    intersection = preds[preds == labels]
    area_intersection = intersection.bincount(minlength=num_classes)

    bincount_preds = preds.bincount(minlength=num_classes)
    bincount_labels = labels.bincount(minlength=num_classes)

    area_union = bincount_preds + bincount_labels - area_intersection

    area_intersection = area_intersection.float().cpu().numpy()
    area_union = area_union.float().cpu().numpy()

    return area_intersection, area_union


def mean_iou(outputs, labels, num_classes=19):
    """ Calculate IoU for torch.Tensor
    Parameters
    ----------
    outputs: torch.Tensor 
        Predicted tensor with size (B*H*W, NumClasses).

    labels: torch.Tensor
        Truth label with size (B*H*W).

    Returns
    -------
    iou: np.float
        Calclated IoU metric.
    """

    preds = outputs.argmax(dim=1)
    labels = labels
    intersection, union = intersection_and_union(preds, labels, num_classes=num_classes)

    return np.mean(intersection / (union + 1e-16))