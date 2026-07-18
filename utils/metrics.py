"""
Segmentation metrics: Dice coefficient and Hausdorff distance.
"""

import numpy as np
from typing import Dict
from medpy.metric.binary import hd95 as medpy_hd95


def dice_score(pred: np.ndarray, label: np.ndarray, num_classes: int = 4) -> Dict:
    """
    Compute Dice coefficient for each structure and mean.

    Labels: 0=background, 1=LVBP, 2=LVM, 3=RV

    Args:
        pred: Prediction volume (D, H, W)
        label: Ground truth volume (D, H, W)

    Returns:
        Dictionary with per-structure and mean Dice
    """
    structure_names = {1: 'lvbp', 2: 'lvm', 3: 'rv'}
    dices = {}

    for c, name in structure_names.items():
        pred_c = (pred == c)
        label_c = (label == c)

        intersection = (pred_c & label_c).sum()
        union = pred_c.sum() + label_c.sum()

        if union == 0:
            dices[name] = 1.0  # Both empty
        else:
            dices[name] = 2.0 * intersection / union

    dices['mean'] = float(np.mean(list(dices.values())))
    return dices


def hausdorff_distance_95(pred: np.ndarray, label: np.ndarray, num_classes: int = 4) -> float:
    """
    Compute 95th percentile Hausdorff distance.

    Args:
        pred: Prediction volume (D, H, W)
        label: Ground truth volume (D, H, W)

    Returns:
        Mean HD95 across all structures (in voxels)
    """
    hd95_values = []

    for c in range(1, num_classes):
        pred_c = (pred == c).astype(np.uint8)
        label_c = (label == c).astype(np.uint8)

        if pred_c.sum() == 0 or label_c.sum() == 0:
            continue

        try:
            hd = medpy_hd95(pred_c, label_c)
            hd95_values.append(hd)
        except Exception:
            continue

    return float(np.mean(hd95_values)) if hd95_values else 0.0


def sensitivity(pred: np.ndarray, label: np.ndarray, cls: int = 1) -> float:
    """Compute sensitivity (recall) for a given class."""
    pred_c = (pred == cls)
    label_c = (label == cls)
    tp = (pred_c & label_c).sum()
    fn = (~pred_c & label_c).sum()
    return tp / (tp + fn + 1e-8)


def specificity(pred: np.ndarray, label: np.ndarray, cls: int = 1) -> float:
    """Compute specificity for a given class."""
    pred_c = (pred == cls)
    label_c = (label == cls)
    tn = (~pred_c & ~label_c).sum()
    fp = (pred_c & ~label_c).sum()
    return tn / (tn + fp + 1e-8)
