"""
Test-Time Augmentation (TTA) strategies for FIDES.

This module implements the discrete-continuous TTA boundary:
- Discrete transformations (flips, 90° rotations): exact symmetries, no interpolation
- Continuous transformations (arbitrary rotations, scale): require interpolation, harmful

Only discrete geometric TTA that matches training augmentation is beneficial.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


def horizontal_flip(x: torch.Tensor) -> torch.Tensor:
    """Horizontal flip (left-right). Exact symmetry of the square grid."""
    return torch.flip(x, dims=[-1])


def vertical_flip(x: torch.Tensor) -> torch.Tensor:
    """Vertical flip (up-down). Exact symmetry of the square grid."""
    return torch.flip(x, dims=[-2])


def rotate_90(x: torch.Tensor) -> torch.Tensor:
    """90° rotation. Exact symmetry of the square grid, no interpolation."""
    return torch.rot90(x, k=1, dims=[-2, -1])


def rotate_continuous(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """
    Continuous rotation by arbitrary angle. Requires interpolation.
    This is NEUTRAL or HARMFUL per TICP — only use for ablation experiments.
    """
    import torchvision.transforms.functional as TF
    angle_rad = float(angle_deg)
    return TF.rotate(x.unsqueeze(0) if x.ndim == 3 else x, angle_rad).squeeze(0) if x.ndim == 3 else TF.rotate(x, angle_rad)


def get_tta_transforms(tta_mode: str) -> List[Tuple[str, callable]]:
    """
    Get the list of TTA transforms for a given mode.

    Args:
        tta_mode: One of 'h', 'hv', 'hvr'
            'h': horizontal flip only
            'hv': horizontal + vertical flip
            'hvr': horizontal + vertical flip + 90° rotation (recommended)

    Returns:
        List of (name, transform) pairs including identity
    """
    transforms = [("identity", lambda x: x)]

    if tta_mode == "h":
        transforms.append(("hflip", horizontal_flip))
    elif tta_mode == "hv":
        transforms.append(("hflip", horizontal_flip))
        transforms.append(("vflip", vertical_flip))
    elif tta_mode == "hvr":
        transforms.append(("hflip", horizontal_flip))
        transforms.append(("vflip", vertical_flip))
        transforms.append(("rot90", rotate_90))
    elif tta_mode is None:
        pass  # No TTA
    else:
        raise ValueError(f"Unknown TTA mode: {tta_mode}")

    return transforms


def apply_tta(
    model: callable,
    image: torch.Tensor,
    tta_mode: str,
) -> torch.Tensor:
    """
    Apply TTA: run model on augmented inputs and average softmax outputs.

    Args:
        model: Model forward function
        image: Input tensor (1, C, H, W)
        tta_mode: TTA mode string

    Returns:
        Averaged softmax output (1, num_classes, H, W)
    """
    transforms_list = get_tta_transforms(tta_mode)
    outputs = []

    for name, transform in transforms_list:
        augmented = transform(image)
        with torch.no_grad():
            logits = model(augmented)
        # Apply inverse transform to logits
        if name == "hflip":
            logits = horizontal_flip(logits)
        elif name == "vflip":
            logits = vertical_flip(logits)
        elif name == "rot90":
            logits = torch.rot90(logits, k=3, dims=[-2, -1])
        outputs.append(F.softmax(logits, dim=1))

    # Average softmax outputs
    return torch.stack(outputs).mean(dim=0)


def apply_continuous_rotation_tta(
    model: callable,
    image: torch.Tensor,
    angles: List[float],
) -> torch.Tensor:
    """
    Continuous rotation TTA (for ablation only). NEUTRAL or HARMFUL per TICP.

    Args:
        model: Model forward function
        image: Input tensor (1, C, H, W)
        angles: List of rotation angles in degrees

    Returns:
        Averaged softmax output
    """
    import torchvision.transforms.functional as TF
    outputs = []

    for angle in angles:
        augmented = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        with torch.no_grad():
            logits = model(augmented)
        logits = TF.rotate(logits, -angle, interpolation=TF.InterpolationMode.BILINEAR)
        outputs.append(F.softmax(logits, dim=1))

    return torch.stack(outputs).mean(dim=0)
