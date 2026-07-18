import numpy as np
from typing import Dict, List


class DiceCalculator:
    """Compute per-class and mean Dice coefficients.

    For C classes (including background), the mean Dice is computed over
    the foreground classes only (classes 1 … C-1), following the standard
    convention in medical image segmentation.
    """

    def __init__(self, num_classes: int = 4, exclude_background: bool = True):
        self.num_classes = num_classes
        self.exclude_background = exclude_background

    def compute(self, pred: np.ndarray, label: np.ndarray) -> float:
        dices = self.compute_per_class(pred, label)
        classes = range(1, self.num_classes) if self.exclude_background else range(self.num_classes)
        vals = [dices[c] for c in classes if dices[c] is not None]
        return float(np.mean(vals)) if vals else 0.0

    def compute_per_class(self, pred: np.ndarray, label: np.ndarray
                          ) -> Dict[int, float]:
        result = {}
        for c in range(self.num_classes):
            pred_c = (pred == c)
            lbl_c = (label == c)
            intersection = np.logical_and(pred_c, lbl_c).sum()
            total = pred_c.sum() + lbl_c.sum()
            if total == 0:
                result[c] = None
            else:
                result[c] = (2.0 * intersection / total)
        return result

    def compute_batch(
        self,
        preds: List[np.ndarray],
        labels: List[np.ndarray],
    ) -> Dict[str, float]:
        all_dices = []
        all_per_class: Dict[int, List[float]] = {c: [] for c in range(1, self.num_classes)}
        for pred, lbl in zip(preds, labels):
            d = self.compute(pred, lbl)
            all_dices.append(d)
            pc = self.compute_per_class(pred, lbl)
            for c in range(1, self.num_classes):
                if pc[c] is not None:
                    all_per_class[c].append(pc[c])

        class_names = {1: 'LVBP', 2: 'LVM', 3: 'RV'}
        result = {
            'mean': float(np.mean(all_dices)),
            'std': float(np.std(all_dices)),
        }
        for c in range(1, self.num_classes):
            name = class_names.get(c, f'Class{c}')
            if all_per_class[c]:
                result[f'{name}_mean'] = float(np.mean(all_per_class[c]))
        return result
