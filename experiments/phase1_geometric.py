"""
Phase I: Geometric Correction (7 configurations)

Incrementally introduces ARP, PSR, LBU, TTA, and morphological refinement
to the broken CorSeg baseline.

Usage:
    python experiments/phase1_geometric.py --acdc_dir /path/to/acdc
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fides.configs import get_configs_by_phase
from fides.pipeline import FIDESPipeline
from data.acdc_loader import ACDCDataset


def main():
    parser = argparse.ArgumentParser(description="Phase I: Geometric Correction")
    parser.add_argument("--acdc_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/phase1")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = ACDCDataset(args.acdc_dir, split="test")
    configs = get_configs_by_phase("I")

    all_results = {}
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Running: {config.name}")
        print(f"{'='*60}")

        pipeline = FIDESPipeline(config, checkpoint_path=args.checkpoint, device=args.device)
        results = pipeline.run(dataset, os.path.join(args.output_dir, config.name))
        all_results[config.name] = results

        print(f"  Dice: {results['mean_dice']:.4f}")
        print(f"  HD95: {results['hd95']:.2f}")

    # Save summary
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    print(f"\n{'='*60}")
    print("Phase I Summary:")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'Dice':<10} {'ΔDice':<10}")
    print("-" * 50)
    baseline = all_results['corseg_original']['mean_dice']
    for name, res in all_results.items():
        delta = res['mean_dice'] - baseline
        print(f"{name:<30} {res['mean_dice']:<10.4f} {delta:+.4f}")


if __name__ == "__main__":
    main()
