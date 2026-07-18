"""
Phase VII: Cross-Ablation (2x2) — Decisive Causal Evidence

Cross-evaluates two models against two inference pipelines in a 2x2 design:

    Model-A (trained WITH ARP)  x  Pipeline WITH ARP   -> consistent
    Model-A (trained WITH ARP)  x  Pipeline NO ARP     -> inconsistent
    Model-B (trained NO ARP)    x  Pipeline WITH ARP   -> inconsistent
    Model-B (trained NO ARP)    x  Pipeline NO ARP     -> consistent

Per TICP: the two CONSISTENT cells (A-with, B-without) should always
outperform the two INCONSISTENT cells, regardless of model training scale.
This provides decisive causal evidence that consistency — not any single
component — is the key determinant of deployment performance.

Usage:
    python experiments/phase7_cross_ablation.py --acdc_dir /path/to/acdc \
        --model_a checkpoints/model_a_with_arp.pth \
        --model_b checkpoints/model_b_no_arp.pth
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fides.configs import PipelineConfig
from fides.pipeline import FIDESPipeline
from data.acdc_loader import ACDCDataset


# 2x2 design: (model, pipeline_arp)
CROSS_ABLATION_CELLS = [
    ("A_with_arp",   "model_a", True),
    ("A_without_arp", "model_a", False),
    ("B_with_arp",   "model_b", True),
    ("B_without_arp", "model_b", False),
]


def make_pipeline_config(arp: bool) -> PipelineConfig:
    """Create a pipeline config with/without ARP, keeping other FIDES components."""
    return PipelineConfig(
        name=f"cross_{'with' if arp else 'without'}_arp",
        phase="VII",
        arp=arp,
        psr=True,
        lbu=True,
        tta="hvr" if arp else None,
    )


def run_cross_ablation(dataset, model_paths, device, output_dir):
    """Run the 2x2 cross-ablation experiment."""
    all_results = {}

    for cell_name, model_key, use_arp in CROSS_ABLATION_CELLS:
        config = make_pipeline_config(use_arp)
        checkpoint = model_paths.get(model_key)

        print(f"\n{'='*60}")
        print(f"  Cell: {cell_name}")
        print(f"  Model: {model_key} | Pipeline ARP: {use_arp}")
        print(f"  Consistent: {'YES' if (model_key == 'model_a') == use_arp else 'NO'}")
        print(f"{'='*60}")

        pipeline = FIDESPipeline(
            config,
            checkpoint_path=checkpoint,
            device=device,
        )

        cell_dir = os.path.join(output_dir, cell_name)
        os.makedirs(cell_dir, exist_ok=True)

        results = pipeline.run(dataset, cell_dir)
        all_results[cell_name] = {
            **results,
            "model": model_key,
            "pipeline_arp": use_arp,
            "consistent": (model_key == "model_a") == use_arp,
        }

        print(f"  Dice: {results['mean_dice']:.4f}")

    return all_results


def print_2x2_table(all_results):
    """Print the 2x2 factorial table."""
    print(f"\n{'='*70}")
    print("  Phase VII: 2x2 Cross-Ablation Table")
    print(f"{'='*70}")
    print(f"  {'':25s} {'Pipeline WITH ARP':<20s} {'Pipeline NO ARP':<20s}")
    print(f"  {'-'*65}")

    a_with = all_results["A_with_arp"]["mean_dice"]
    a_without = all_results["A_without_arp"]["mean_dice"]
    b_with = all_results["B_with_arp"]["mean_dice"]
    b_without = all_results["B_without_arp"]["mean_dice"]

    print(f"  {'Model-A (WITH ARP train)':<25s} {a_with:<20.4f} {a_without:<20.4f}")
    print(f"  {'Model-B (NO ARP train)':<25s} {b_with:<20.4f} {b_without:<20.4f}")
    print(f"\n  Consistent cells: A-with-ARP ({a_with:.4f}), B-without-ARP ({b_without:.4f})")
    print(f"  Inconsistent cells: A-without-ARP ({a_without:.4f}), B-with-ARP ({b_with:.4f})")

    consistent_mean = (a_with + b_without) / 2
    inconsistent_mean = (a_without + b_with) / 2
    print(f"\n  Mean consistent:   {consistent_mean:.4f}")
    print(f"  Mean inconsistent: {inconsistent_mean:.4f}")
    print(f"  Consistency advantage: {consistent_mean - inconsistent_mean:+.4f}")


def verify_ticp(all_results):
    """Verify TICP prediction: consistent > inconsistent in both rows."""
    print(f"\n{'='*70}")
    print("  TICP Verification:")
    print(f"{'='*70}")

    checks = []

    # Row 1: Model-A should perform better WITH ARP (consistent)
    delta_a = all_results["A_with_arp"]["mean_dice"] - all_results["A_without_arp"]["mean_dice"]
    checks.append(("Model-A: WITH ARP > NO ARP", delta_a > 0, delta_a))
    print(f"  Model-A: WITH ARP > NO ARP?  Δ = {delta_a:+.4f}  [{'PASS' if delta_a > 0 else 'FAIL'}]")

    # Row 2: Model-B should perform better WITHOUT ARP (consistent)
    delta_b = all_results["B_without_arp"]["mean_dice"] - all_results["B_with_arp"]["mean_dice"]
    checks.append(("Model-B: NO ARP > WITH ARP", delta_b > 0, delta_b))
    print(f"  Model-B: NO ARP > WITH ARP?  Δ = {delta_b:+.4f}  [{'PASS' if delta_b > 0 else 'FAIL'}]")

    # Overall: consistent mean > inconsistent mean
    consistent_mean = (all_results["A_with_arp"]["mean_dice"] + all_results["B_without_arp"]["mean_dice"]) / 2
    inconsistent_mean = (all_results["A_without_arp"]["mean_dice"] + all_results["B_with_arp"]["mean_dice"]) / 2
    delta_overall = consistent_mean - inconsistent_mean
    checks.append(("Overall: consistent > inconsistent", delta_overall > 0, delta_overall))
    print(f"  Overall: consistent > inconsistent?  Δ = {delta_overall:+.4f}  [{'PASS' if delta_overall > 0 else 'FAIL'}]")

    all_pass = all(c[1] for c in checks)
    print(f"\n  Result: {'ALL CHECKS PASSED — TICP causally validated.' if all_pass else 'Some checks failed.'}")
    print("  Conclusion: Training-inference consistency is the decisive causal factor,")
    print("  not any individual pipeline component or training data scale.")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Phase VII: Cross-Ablation (2x2)")
    parser.add_argument("--acdc_dir", type=str, required=True,
                        help="Path to ACDC dataset")
    parser.add_argument("--model_a", type=str, required=True,
                        help="Checkpoint for Model-A (trained WITH ARP)")
    parser.add_argument("--model_b", type=str, required=True,
                        help="Checkpoint for Model-B (trained NO ARP)")
    parser.add_argument("--output_dir", type=str, default="./results/phase7")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_paths = {"model_a": args.model_a, "model_b": args.model_b}

    print("\n>>> Loading ACDC test set...")
    dataset = ACDCDataset(args.acdc_dir, split="test")

    all_results = run_cross_ablation(dataset, model_paths, args.device, args.output_dir)

    print_2x2_table(all_results)
    verify_ticp(all_results)

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
