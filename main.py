"""FIDES: Fidelity-Informed Deployment for Exact Segmentation.

One-click pipeline comparison demonstrating that inference pipeline design
(not model architecture) is the decisive factor in deployment performance.

Usage:
    python main.py
    python main.py --n-samples 100 --target-size 96
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_simulator import CardiacMRISimulator
from models.mednext import SimulatedMedNeXt
from pipelines import (
    DefectivePipeline,
    ARPOnlyPipeline,
    LBUOnlyPipeline,
    ARPLBUPipeline,
)
from evaluation.metrics import DiceCalculator
from utils.visualization import visualize_comparison


def run_experiment(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    simulator = CardiacMRISimulator(
        variable_size=True,
        size_range=(args.size_min, args.size_max),
        noise_std=args.noise_std,
    )
    images, labels = simulator.generate(n_samples=args.n_samples)
    print(f"Generated {args.n_samples} cardiac MRI samples "
          f"(size range: {args.size_min}-{args.size_max})")

    model = SimulatedMedNeXt(
        oracle=True,
        num_classes=4,
        logit_scale=args.logit_scale,
        blur_sigma=args.blur_sigma,
        padding_scale=args.padding_scale,
        non_arp_logit_scale=args.non_arp_logit_scale,
        non_arp_extra_blur=args.non_arp_extra_blur,
        distortion_blur_factor=args.distortion_blur_factor,
        non_arp_noise_std=args.non_arp_noise_std,
    )
    model.eval()

    pipelines = [
        ('Defective', DefectivePipeline()),
        ('ARP-only',  ARPOnlyPipeline()),
        ('LBU-only',  LBUOnlyPipeline()),
        ('ARP+LBU',   ARPLBUPipeline()),
    ]

    all_preds = {}
    all_dices = {}
    all_per_class = {}

    for name, pipeline in pipelines:
        preds = []
        dices = []
        per_class_dices = {c: [] for c in range(1, 4)}
        for img, lbl in zip(images, labels):
            img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
            lbl_t = torch.from_numpy(lbl).unsqueeze(0).long()
            with torch.no_grad():
                pred = pipeline.run(img_t, lbl_t, model, args.target_size)
            pred_np = pred.squeeze(0).numpy()

            calculator = DiceCalculator(num_classes=4)
            per_class = calculator.compute_per_class(pred_np, lbl)
            dice = calculator.compute(pred_np, lbl)
            preds.append(pred_np)
            dices.append(dice)
            for c in range(1, 4):
                if per_class[c] is not None:
                    per_class_dices[c].append(per_class[c])

        all_preds[name] = preds
        all_dices[name] = dices
        all_per_class[name] = {
            c: np.mean(per_class_dices[c]) if per_class_dices[c] else 0.0
            for c in range(1, 4)
        }

    print("\n" + "=" * 70)
    print("  FIDES Pipeline Comparison Results")
    print("=" * 70)
    print(f"  {'Pipeline':<12} {'Dice (mean +/- std)':<22} {'vs Defective':<15}")
    print("-" * 70)

    baseline = np.mean(all_dices['Defective'])
    for name, _ in pipelines:
        m = np.mean(all_dices[name])
        s = np.std(all_dices[name])
        delta = m - baseline
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"  {name:<12} {m:.4f} +/- {s:.4f}       {delta_str}")

    print(f"\n  Per-class Dice (LVBP / LVM / RV):")
    print("-" * 70)
    for name, _ in pipelines:
        pc = all_per_class[name]
        print(f"  {name:<12}  LVBP={pc[1]:.3f}  LVM={pc[2]:.3f}  RV={pc[3]:.3f}")

    d_def = np.mean(all_dices['Defective'])
    d_arp = np.mean(all_dices['ARP-only'])
    d_lbu = np.mean(all_dices['LBU-only'])
    d_al  = np.mean(all_dices['ARP+LBU'])

    interaction = (d_al - d_lbu) - (d_arp - d_def)

    print(f"\n  2x2 Factorial Analysis (ARP x LBU)")
    print(f"  {'Cell':<20} {'Mean Dice':<12}")
    print(f"  {'-'*32}")
    print(f"  {'No ARP, No LBU':<20} {d_def:.4f}")
    print(f"  {'ARP, No LBU':<20} {d_arp:.4f}")
    print(f"  {'No ARP, LBU':<20} {d_lbu:.4f}")
    print(f"  {'ARP, LBU':<20} {d_al:.4f}")
    print(f"\n  Main effect ARP  = {(d_arp + d_al)/2 - (d_def + d_lbu)/2:+.4f}")
    print(f"  Main effect LBU  = {(d_lbu + d_al)/2 - (d_def + d_arp)/2:+.4f}")
    print(f"  Interaction      = {interaction:+.4f}", end="")

    if interaction < -0.1:
        print("  (strong negative -> functional substitutes)")
    elif interaction < 0:
        print("  (negative -> partial substitutes)")
    elif interaction < 0.1:
        print("  (near zero -> independent)")
    else:
        print("  (positive -> complementary)")

    print(f"\n  LBU effect without ARP = {d_lbu - d_def:+.4f}")
    print(f"  LBU effect with ARP    = {d_al - d_arp:+.4f}")
    print(f"  ARP effect without LBU = {d_arp - d_def:+.4f}")
    print(f"  ARP effect with LBU    = {d_al - d_lbu:+.4f}")

    ordering_ok = d_def < d_lbu < d_arp < d_al
    interaction_ok = interaction < 0
    print(f"\n  Target pattern: Defective < LBU-only < ARP-only < ARP+LBU")
    print(f"  Pattern achieved: {'YES' if ordering_ok else 'NO'}")
    print(f"  Negative interaction: {'YES' if interaction_ok else 'NO'}")

    visualize_comparison(images, labels, all_preds, all_dices,
                         output_dir=args.output, n_show=3)
    print(f"\nResults saved to {args.output}/")


def main():
    parser = argparse.ArgumentParser(
        description='FIDES: Pipeline Comparison for Cardiac MRI Segmentation')
    parser.add_argument('--n-samples', type=int, default=50)
    parser.add_argument('--target-size', type=int, default=96)
    parser.add_argument('--size-min', type=int, default=100)
    parser.add_argument('--size-max', type=int, default=400)
    parser.add_argument('--noise-std', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--logit-scale', type=float, default=10.0)
    parser.add_argument('--blur-sigma', type=float, default=0.3)
    parser.add_argument('--padding-scale', type=float, default=0.5)
    parser.add_argument('--non-arp-logit-scale', type=float, default=10.0)
    parser.add_argument('--non-arp-extra-blur', type=float, default=0.5)
    parser.add_argument('--distortion-blur-factor', type=float, default=1.8)
    parser.add_argument('--non-arp-noise-std', type=float, default=2.8)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
