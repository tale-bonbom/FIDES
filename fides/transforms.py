"""
Core transforms for the FIDES pipeline.

This module implements the four key transforms derived from TICP:
1. Aspect Ratio Preservation (ARP) - geometric consistency
2. Pixel Spacing Restoration (PSR) - geometric consistency
3. Logits-Bilinear-Upsampling (LBU) - post-processing consistency
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def aspect_ratio_preservation(
    image: np.ndarray,
    target_size: int = 224,
) -> Tuple[np.ndarray, dict]:
    """
    Aspect Ratio Preservation (ARP): Pad the image to a square before resizing.

    This ensures the resize operation is isotropic, preserving the physical
    aspect ratio of cardiac structures. Without ARP, non-isotropic resizing
    introduces systematic spatial distortion.

    Args:
        image: Input image array (H, W) or (H, W, C)
        target_size: Target size for the square output (default: 224)

    Returns:
        Padded and resized image, metadata for inverse transform
    """
    if image.ndim == 2:
        H, W = image.shape
    else:
        H, W = image.shape[:2]

    # Pad to square using the longer dimension
    if H > W:
        pad_h, pad_w = 0, (H - W) // 2
        pad_w_extra = (H - W) % 2
        if image.ndim == 2:
            padded = np.pad(image, ((0, 0), (pad_w, pad_w + pad_w_extra)), mode='constant')
        else:
            padded = np.pad(image, ((0, 0), (pad_w, pad_w + pad_w_extra), (0, 0)), mode='constant')
    elif W > H:
        pad_h, pad_w = (W - H) // 2, 0
        pad_h_extra = (W - H) % 2
        if image.ndim == 2:
            padded = np.pad(image, ((pad_h, pad_h + pad_h_extra), (0, 0)), mode='constant')
        else:
            padded = np.pad(image, ((pad_h, pad_h + pad_h_extra), (0, 0), (0, 0)), mode='constant')
    else:
        padded = image.copy()

    # Resize to target size using bilinear interpolation
    if padded.ndim == 2:
        padded_4d = padded[None, None, ...].astype(np.float32)
    else:
        padded_4d = padded[None, ...].transpose(0, 3, 1, 2).astype(np.float32)

    tensor = torch.from_numpy(padded_4d)
    resized = F.interpolate(tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
    result = resized.numpy()

    if image.ndim == 2:
        result = result[0, 0]
    else:
        result = result[0].transpose(1, 2, 0)

    metadata = {
        'orig_h': H,
        'orig_w': W,
        'pad_h': pad_h if H > W else (pad_h if W > H else 0),
        'pad_w': pad_w if W > H else (pad_w if H > W else 0),
        'pad_h_extra': pad_h_extra if 'pad_h_extra' in dir() else 0,
        'pad_w_extra': pad_w_extra if 'pad_w_extra' in dir() else 0,
        'target_size': target_size,
        'padded_size': max(H, W),
    }

    return result, metadata


def pixel_spacing_restoration(
    logits: torch.Tensor,
    orig_h: int,
    orig_w: int,
    pad_metadata: dict,
) -> torch.Tensor:
    """
    Pixel Spacing Restoration (PSR): Resample prediction back to original spacing.

    After model prediction, the output is resampled back to the original pixel
    spacing using bilinear interpolation, restoring the physical dimensions
    of the segmentation.

    Args:
        logits: Model output logits (C, H, W) or (B, C, H, W)
        orig_h: Original image height
        orig_w: Original image width
        pad_metadata: Metadata from ARP step

    Returns:
        Resampled logits at original resolution
    """
    padded_size = pad_metadata['padded_size']

    # First resize back to padded size
    if logits.ndim == 3:
        logits_4d = logits[None]
    else:
        logits_4d = logits

    # Resize to padded square size
    resized = F.interpolate(
        logits_4d, size=(padded_size, padded_size),
        mode='bilinear', align_corners=False
    )

    # Remove padding to restore original dimensions
    pad_h = pad_metadata.get('pad_h', 0)
    pad_w = pad_metadata.get('pad_w', 0)
    pad_h_extra = pad_metadata.get('pad_h_extra', 0)
    pad_w_extra = pad_metadata.get('pad_w_extra', 0)

    if pad_h > 0 or pad_h_extra > 0:
        resized = resized[:, :, pad_h:pad_h + orig_h, :]
    if pad_w > 0 or pad_w_extra > 0:
        resized = resized[:, :, :, pad_w:pad_w + orig_w]

    if logits.ndim == 3:
        return resized[0]
    return resized


def logits_bilinear_upsampling(
    logits: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """
    Logits-Bilinear-Upsampling (LBU): Upsample continuous logits before argmax.

    Rather than applying argmax before upsampling (which produces discrete
    "mosaic" boundaries through nearest-neighbor interpolation), LBU preserves
    the continuous probability maps during upsampling and applies argmax only
    at the final step.

    Args:
        logits: Continuous model output (C, H, W) or (B, C, H, W)
        target_h: Target height
        target_w: Target width

    Returns:
        Upsampled logits at target resolution
    """
    if logits.ndim == 3:
        logits_4d = logits[None]
    else:
        logits_4d = logits

    upsampled = F.interpolate(
        logits_4d, size=(target_h, target_w),
        mode='bilinear', align_corners=False
    )

    if logits.ndim == 3:
        return upsampled[0]
    return upsampled


def z_score_normalize(image: np.ndarray, mean: float = None, std: float = None) -> np.ndarray:
    """Z-score normalization matching training conditions."""
    if mean is None:
        mean = image.mean()
    if std is None:
        std = image.std()
    if std == 0:
        std = 1.0
    return (image.astype(np.float32) - mean) / std


def apply_normalization(image: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Apply normalization method."""
    if method == "zscore":
        return z_score_normalize(image)
    elif method == "pct":
        p1, p99 = np.percentile(image, [1, 99])
        if p99 - p1 == 0:
            return image.astype(np.float32)
        return np.clip((image.astype(np.float32) - p1) / (p99 - p1), 0, 1)
    elif method == "clahe":
        try:
            from skimage import exposure
            return exposure.equalize_adapthist(image.astype(np.float32), clip_limit=0.03)
        except ImportError:
            return z_score_normalize(image)
    elif method == "none":
        return image.astype(np.float32)
    else:
        return z_score_normalize(image)
