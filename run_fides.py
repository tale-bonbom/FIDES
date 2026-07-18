#!/usr/bin/env python3
"""
FIDES: Fidelity-Informed Deployment for Exact Segmentation
Main entry point for running FIDES pipeline configurations.

Usage:
    python run_fides.py --config fides_optimal --input /path/to/acdc --output ./results
    python run_fides.py --config corseg_original --input /path/to/acdc --output ./results
    python run_fides.py --config all --input /path/to/acdc --output ./results
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

from fides.configs import get_config, CONFIG_NAMES
from fides.pipeline import FIDESPipeline
from data.acdc_loader import ACDCDataset


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="FIDES: Fidelity-Informed Deployment for Exact Segmentation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="fides_optimal",
        choices=CONFIG_NAMES + ["all"],
        help="Pipeline configuration name (default: fides_optimal)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to ACDC dataset directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for predictions (default: ./results)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/mednext_l.pth",
        help="Path to model checkpoint (default: ./checkpoints/mednext_l.pth). "
            "Download from https://github.com/tale-bonbom/FIDES/releases/tag/v1.0.0",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def run_single_config(config_name, args):
    """Run a single pipeline configuration."""
    print(f"\n{'='*60}")
    print(f"Running configuration: {config_name}")
    print(f"{'='*60}\n")

    config = get_config(config_name)
    config.print_summary()

    # Load dataset
    dataset = ACDCDataset(args.input, split="test")

    # Initialize pipeline
    pipeline = FIDESPipeline(config, checkpoint_path=args.checkpoint, device=args.device)

    # Run inference
    output_dir = os.path.join(args.output, config_name)
    os.makedirs(output_dir, exist_ok=True)

    results = pipeline.run(dataset, output_dir)

    # Print summary
    print(f"\nResults for {config_name}:")
    print(f"  Mean Dice: {results['mean_dice']:.4f}")
    print(f"  LVBP Dice: {results['dice_lvbp']:.4f}")
    print(f"  LVM Dice:  {results['dice_lvm']:.4f}")
    print(f"  RV Dice:   {results['dice_rv']:.4f}")
    print(f"  HD95:      {results['hd95']:.2f} mm")

    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    print(f"FIDES v1.0.0 | Random seed: {args.seed} | Device: {args.device}")

    if args.config == "all":
        all_results = {}
        for config_name in CONFIG_NAMES:
            all_results[config_name] = run_single_config(config_name, args)

        print(f"\n{'='*60}")
        print("All configurations completed. Summary:")
        print(f"{'='*60}\n")
        for name, res in all_results.items():
            print(f"  {name:40s} Dice: {res['mean_dice']:.4f}")
    else:
        run_single_config(args.config, args)


if __name__ == "__main__":
    main()
