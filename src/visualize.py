from typing import List, Tuple

import numpy as np


def apply_mask_on_image(
    mask: np.ndarray, image: np.ndarray, cmaps: List[Tuple[str, Tuple[int]]],
    ignore_index: List[int] = []
) -> np.ndarray:
    """Apply color maps on original image.
    Args:
        mask (np.ndarray): Mask image with shape (num_classes, H, W).
        image (np.ndarray): Original image with shape (H, W, C).
        cmaps (list): A list of tuple like
                      [('class_name_1', (0, 128, 128)), (), ...].
    Returns:
        np.ndarray: Visualized image using specified color maps with shape
                    (Height, Width, 3).
    """
    result_image: np.ndarray = np.copy(image)

    img_2d: np.ndarray = np.argmax(mask, axis=0)
    for class_idx, (_, cmap) in enumerate(cmaps):
        if class_idx in ignore_index:
            continue
        # points: [[x, y], [x, y], ...]
        points: np.ndarray = np.argwhere(img_2d == class_idx)
        for x, y in points:
            result_image[x, y, :] = cmap

    return result_image


def visualize_errors(
    pred_mask: np.ndarray, gt_mask: np.ndarray, image: np.ndarray,
    cmaps: List[Tuple[str, Tuple[int]]], error_type: str = 'fn',
    ignore_index: List[int] = []
) -> np.ndarray:
    """Visualize false positives or false negatives of given image.
    Args:
        pred_mask (np.ndarray): Mask image with shape (num_classes, H, W).
        gt_mask (np.ndarray): Ground truth mas kwith shape (H, W).
        image (np.ndarray): Original image with shape (H, W, C).
        cmaps (list): A list of tuple like
                      [('class_name_1', (0, 128, 128)), (), ...].
        error_type (str): 'fp' or 'fn'.
    Returns:
        np.ndarray: Visualized image using specified color maps with shape
                    (Height, Width, 3).
    """
    if error_type not in ('fp', 'fn'):
        raise ValueError('error_type must be "fp" or "fn".')
    result_image: np.ndarray = np.copy(image)

    pred_mask_2d: np.ndarray = np.argmax(pred_mask, axis=0)
    for class_idx, (_, cmap) in enumerate(cmaps):
        if class_idx in ignore_index:
            continue
        # points: [[x, y], [x, y], ...]
        pred_points: np.ndarray = np.argwhere(pred_mask_2d == class_idx)
        gt_points: np.ndarray = np.argwhere(gt_mask == class_idx)
        err_points = []
        if error_type == 'fp':
            for pred in pred_points:
                if pred not in gt_points:
                    err_points.append(pred)
        else:
            for gt in gt_points:
                if gt not in pred_points:
                    err_points.append(gt)

        err_points = np.array(err_points)
        for x, y in err_points:
            result_image[x, y, :] = cmap

    return result_image
