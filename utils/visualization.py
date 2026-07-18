"""
Visualization utilities for FIDES results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_dice_comparison(results: Dict[str, Dict], save_path: str = None):
    """Plot Dice comparison across configurations."""
    names = list(results.keys())
    dices = [results[n]['mean_dice'] for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, dices, color=['red' if d < 0.5 else 'green' for d in dices])
    ax.set_ylabel('Mean 3D Dice')
    ax.set_title('Pipeline Configuration Comparison')
    ax.set_xticklabels(names, rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_prediction_overlay(
    image: np.ndarray,
    pred: np.ndarray,
    label: np.ndarray,
    slice_idx: int = 0,
    save_path: str = None,
):
    """Overlay prediction and ground truth on the image."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image[slice_idx], cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(label[slice_idx], cmap='jet')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred[slice_idx], cmap='jet')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_bland_altman(pred_values, true_values, metric_name: str, save_path: str = None):
    """Generate Bland-Altman plot."""
    mean_vals = (np.array(pred_values) + np.array(true_values)) / 2
    diff_vals = np.array(pred_values) - np.array(true_values)
    bias = np.mean(diff_vals)
    loa = 1.96 * np.std(diff_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mean_vals, diff_vals, alpha=0.6, c='blue')
    ax.axhline(bias, color='red', linestyle='--', label=f'Bias = {bias:.2f}')
    ax.axhline(bias + loa, color='gray', linestyle=':', label=f'+1.96 SD = {bias+loa:.2f}')
    ax.axhline(bias - loa, color='gray', linestyle=':', label=f'-1.96 SD = {bias-loa:.2f}')
    ax.set_xlabel(f'Mean (Pred + True) / 2')
    ax.set_ylabel(f'Pred - True')
    ax.set_title(f'Bland-Altman: {metric_name}')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
