"""
Phase V: Inference Strategy Boundary (8 configurations)

Tests continuous rotations (±5°, ±10°), scale augmentation, logits smoothing,
and temperature scaling.
Per TICP: only discrete 90° rotations are beneficial; continuous is neutral/harmful.

Usage:
    python experiments/phase5_strategy.py --acdc_dir /path/to/acdc
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from fides.configs import get_configs_by_phase
from fides.pipeline import FIDESPipeline
from data.acdc_loader import ACDCDataset

def main():
    parser = argparse.ArgumentParser(description="Phase V: Inference Strategy Boundary")
    parser.add_argument("--acdc_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/phase5")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = ACDCDataset(args.acdc_dir, split="test")
    configs = get_configs_by_phase("V")

    all_results = {}
    for config in configs:
        print(f"\nRunning: {config.name}")
        pipeline = FIDESPipeline(config, checkpoint_path=args.checkpoint, device=args.device)
        results = pipeline.run(dataset, os.path.join(args.output_dir, config.name))
        all_results[config.name] = results
        print(f"  Dice: {results['mean_dice']:.4f}")

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nPhase V Summary: All continuous strategies should be neutral or slightly harmful.")

if __name__ == "__main__":
    main()
