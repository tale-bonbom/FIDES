"""
Post-processing strategies for ablation (Phase II).

Per TICP Design Principle 3: ALL post-processing is neutral or harmful
for well-trained models. These implementations are provided for ablation
experiments to verify this prediction.
"""

import torch
import numpy as np
from typing import Optional
from scipy import ndimage
from skimage import measure, morphology


def connected_component_filter(
    pred: np.ndarray,
    num_classes: int = 4,
    min_size: int = 50,
) -> np.ndarray:
    """
    Connected Component Filtering (CCF): Remove small isolated regions.

    Per TICP: NEUTRAL or HARMFUL for well-trained models.
    """
    filtered = np.zeros_like(pred)
    for c in range(1, num_classes):
        mask = (pred == c).astype(np.uint8)
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            for i, s in enumerate(sizes):
                if s >= min_size:
                    filtered[labeled == i + 1] = c
    return filtered


def iterative_surface_fitting(
    pred: np.ndarray,
    num_classes: int = 4,
    iterations: int = 3,
) -> np.ndarray:
    """
    Iterative Surface Fitting (ISF): Smooth boundaries using surface fitting.

    Per TICP: HARMFUL for well-trained models.
    """
    result = pred.copy()
    for _ in range(iterations):
        for c in range(1, num_classes):
            mask = (result == c).astype(np.float32)
            smoothed = ndimage.gaussian_filter(mask, sigma=1.0)
            result[smoothed > 0.5] = c
    return result


def adaptive_morphological_filter(
    pred: np.ndarray,
    num_classes: int = 4,
) -> np.ndarray:
    """
    Adaptive Morphological Filtering (AMF): Apply morphological operations.

    Per TICP: HARMFUL for well-trained models.
    """
    result = pred.copy()
    for c in range(1, num_classes):
        mask = (result == c).astype(np.uint8)
        # Opening then closing
        mask = morphology.binary_opening(mask, morphology.disk(1))
        mask = morphology.binary_closing(mask, morphology.disk(1))
        result[mask] = c
    return result


def confidence_based_removal(
    logits: np.ndarray,
    threshold: float = 0.5,
    num_classes: int = 4,
) -> np.ndarray:
    """
    Confidence-Based Removal (CBR): Remove low-confidence predictions.

    Per TICP: HARMFUL for well-trained models.

    Args:
        logits: Softmax probabilities (C, H, W)
        threshold: Confidence threshold
    """
    pred = np.argmax(logits, axis=0)
    confidence = np.max(logits, axis=0)
    pred[confidence < threshold] = 0  # Background
    return pred


def roi_pruning(
    pred: np.ndarray,
    num_classes: int = 4,
) -> np.ndarray:
    """
    Region-of-Interest Pruning: Remove predictions outside the cardiac region.

    Per TICP: HARMFUL for well-trained models.
    """
    result = pred.copy()
    # Find the bounding box of all non-background pixels
    non_bg = pred > 0
    if non_bg.sum() == 0:
        return result

    rows = np.any(non_bg, axis=1)
    cols = np.any(non_bg, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Prune isolated regions outside the main bounding box
    for c in range(1, num_classes):
        mask = (pred == c).astype(np.uint8)
        labeled, num_features = ndimage.label(mask)
        for i in range(1, num_features + 1):
            region = labeled == i
            rows_r = np.any(region, axis=1)
            cols_r = np.any(region, axis=0)
            rmin_r, rmax_r = np.where(rows_r)[0][[0, -1]]
            cmin_r, cmax_r = np.where(cols_r)[0][[0, -1]]
            # If region is far from main ROI, remove it
            if (rmin_r > rmax + 10 or rmax_r < rmin - 10 or
                cmin_r > cmax + 10 or cmax_r < cmin - 10):
                result[region] = 0
    return result


def apply_postprocessing(
    pred: np.ndarray,
    method: str,
    logits: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply post-processing method to prediction.

    Args:
        pred: Argmax prediction (H, W)
        method: Post-processing method name
        logits: Optional softmax logits (for confidence-based methods)

    Returns:
        Post-processed prediction
    """
    if method is None or method == "none":
        return pred

    handlers = {
        "ccf": connected_component_filter,
        "isf": iterative_surface_fitting,
        "amf": adaptive_morphological_filter,
        "cbr": lambda p: confidence_based_removal(logits) if logits is not None else p,
        "roi": roi_pruning,
        "morph": lambda p: adaptive_morphological_filter(p),
        "selective_ccf": lambda p: connected_component_filter(p, min_size=100),
        "ccf_isf": lambda p: iterative_surface_fitting(connected_component_filter(p)),
        "full": lambda p: roi_pruning(iterative_surface_fitting(connected_component_filter(p))),
    }

    handler = handlers.get(method)
    if handler is None:
        raise ValueError(f"Unknown post-processing method: {method}")

    return handler(pred)
