"""
Phase IV: Input Perturbation (8 configurations)

Tests PCT normalization, CLAHE, intensity TTA (iTTA), and combinations.
Per TICP: input perturbations are neutral or harmful.

Usage:
    python experiments/phase4_perturbation.py --acdc_dir /path/to/acdc
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from fides.configs import get_configs_by_phase
from fides.pipeline import FIDESPipeline
from data.acdc_loader import ACDCDataset

def main():
    parser = argparse.ArgumentParser(description="Phase IV: Input Perturbation")
    parser.add_argument("--acdc_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/phase4")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = ACDCDataset(args.acdc_dir, split="test")
    configs = get_configs_by_phase("IV")

    all_results = {}
    for config in configs:
        print(f"\nRunning: {config.name}")
        pipeline = FIDESPipeline(config, checkpoint_path=args.checkpoint, device=args.device)
        results = pipeline.run(dataset, os.path.join(args.output_dir, config.name))
        all_results[config.name] = results
        print(f"  Dice: {results['mean_dice']:.4f}")

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nPhase IV Summary: PCT should be catastrophic; CLAHE should be neutral.")

if __name__ == "__main__":
    main()
