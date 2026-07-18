"""
Phase II: Post-Processing Disqualification (7 configurations)

Tests CCF, ISF, AMF, CBR, ROI pruning, and combinations.
Per TICP: ALL post-processing is neutral or harmful.

Usage:
    python experiments/phase2_postprocessing.py --acdc_dir /path/to/acdc
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from fides.configs import get_configs_by_phase
from fides.pipeline import FIDESPipeline
from data.acdc_loader import ACDCDataset

def main():
    parser = argparse.ArgumentParser(description="Phase II: Post-Processing Disqualification")
    parser.add_argument("--acdc_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/phase2")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = ACDCDataset(args.acdc_dir, split="test")
    configs = get_configs_by_phase("II")

    all_results = {}
    for config in configs:
        print(f"\nRunning: {config.name}")
        pipeline = FIDESPipeline(config, checkpoint_path=args.checkpoint, device=args.device)
        results = pipeline.run(dataset, os.path.join(args.output_dir, config.name))
        all_results[config.name] = results
        print(f"  Dice: {results['mean_dice']:.4f}  ΔDice: {results['mean_dice'] - all_results.get('ccf', {}).get('mean_dice', results['mean_dice']):+.4f}")

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nPhase II Summary: All post-processing should show neutral or negative ΔDice")

if __name__ == "__main__":
    main()
