"""
Result table generation for FIDES paper.

Generates LaTeX and CSV tables for:
- Table 1: Phase I-V pipeline configuration results
- Table 2: Phase VI cross-architecture results
- Table 3: Clinical metrics (EDV, ESV, EF)
- Table 4: Phase VII 2x2 cross-ablation

Usage:
    python clinical/tables.py --results_dir ./results --output_dir ./tables
"""

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir: str) -> Dict:
    """Load all phase results from the results directory."""
    results = {}
    for phase_dir in sorted(os.listdir(results_dir)):
        phase_path = os.path.join(results_dir, phase_dir)
        if not os.path.isdir(phase_path):
            continue
        results_file = os.path.join(phase_path, "results.json")
        if os.path.exists(results_file):
            with open(results_file) as f:
                results[phase_dir] = json.load(f)
    return results


def generate_table1_phases(results: Dict) -> str:
    """
    Table 1: Phase I-V pipeline configuration results.

    Columns: Config, Dice, LVBP, LVM, RV, HD95, ΔDice
    """
    rows = []
    for phase_name in sorted(results.keys()):
        phase_results = results[phase_name]
        baseline_dice = None
        for config_name in sorted(phase_results.keys()):
            r = phase_results[config_name]
            if baseline_dice is None:
                baseline_dice = r["mean_dice"]
            delta = r["mean_dice"] - baseline_dice
            rows.append(
                f"  {config_name} & {r['mean_dice']:.4f} & {r.get('dice_lvbp', 0):.4f} & "
                f"{r.get('dice_lvm', 0):.4f} & {r.get('dice_rv', 0):.4f} & "
                f"{r.get('hd95', 0):.2f} & {delta:+.4f} \\\\"
            )

    latex = """\\begin{table}[t]
\\centering
\\caption{Pipeline configuration results across Phases I-V (38 configurations).}
\\label{tab:phase_results}
\\begin{tabular}{lcccccc}
\\toprule
Configuration & Dice & LVBP & LVM & RV & HD95 (mm) & $\\Delta$Dice \\\\
\\midrule
"""
    latex += "\n".join(rows) + "\n"
    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    return latex


def generate_table2_cross_arch(results: Dict) -> str:
    """
    Table 2: Phase VI cross-architecture results.

    Rows: Architecture, Columns: FIDES-Optimal, CorSeg-Original, ΔDice
    """
    rows = []
    phase6 = results.get("phase6", {})
    acdc = phase6.get("acdc_results", phase6)

    for arch in sorted(acdc.keys()):
        pipelines = acdc[arch]
        fides = pipelines.get("fides_optimal", {}).get("mean_dice", 0)
        corseg = pipelines.get("corseg_original", {}).get("mean_dice", 0)
        delta = fides - corseg
        rows.append(f"  {arch} & {fides:.4f} & {corseg:.4f} & {delta:+.4f} \\\\")

    latex = """\\begin{table}[t]
\\centering
\\caption{Cross-architecture validation (Phase VI). FIDES-Optimal vs. CorSeg-Original.}
\\label{tab:cross_arch}
\\begin{tabular}{lccc}
\\toprule
Architecture & FIDES-Optimal & CorSeg-Original & $\\Delta$Dice \\\\
\\midrule
"""
    latex += "\n".join(rows) + "\n"
    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    return latex


def generate_table3_clinical(stats: Dict) -> str:
    """
    Table 3: Clinical metrics (EDV, ESV, EF).

    Rows: Metric, Columns: ICC, MAE, Bias, LoA
    """
    rows = []
    for metric in ["EDV", "ESV", "EF"]:
        if metric not in stats:
            continue
        s = stats[metric]
        icc_val = s["icc"]["icc"]
        mae = s["mae"]
        ba = s["bland_altman"]
        rows.append(
            f"  {metric} & {icc_val:.3f} & {mae:.2f} & "
            f"{ba['mean_bias']:.2f} & [{ba['loa_lower']:.2f}, {ba['loa_upper']:.2f}] \\\\"
        )

    latex = """\\begin{table}[t]
\\centering
\\caption{Clinical metric agreement between FIDES and ground truth.}
\\label{tab:clinical}
\\begin{tabular}{lcccc}
\\toprule
Metric & ICC(2,1) & MAE & Bias & LoA (95\\%) \\\\
\\midrule
"""
    latex += "\n".join(rows) + "\n"
    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    return latex


def generate_table4_cross_ablation(results: Dict) -> str:
    """
    Table 4: Phase VII 2x2 cross-ablation.

    Rows: Model-A/B, Columns: Pipeline WITH/NO ARP
    """
    phase7 = results.get("phase7", {})

    a_with = phase7.get("A_with_arp", {}).get("mean_dice", 0)
    a_without = phase7.get("A_without_arp", {}).get("mean_dice", 0)
    b_with = phase7.get("B_with_arp", {}).get("mean_dice", 0)
    b_without = phase7.get("B_without_arp", {}).get("mean_dice", 0)

    latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Phase VII cross-ablation (2$\\times$2). Consistent cells are highlighted.}}
\\label{{tab:cross_ablation}}
\\begin{{tabular}}{{lcc}}
\\toprule
& Pipeline WITH ARP & Pipeline NO ARP \\\\
\\midrule
Model-A (WITH ARP train) & \\textbf{{{a_with:.4f}}} & {a_without:.4f} \\\\
Model-B (NO ARP train)    & {b_with:.4f} & \\textbf{{{b_without:.4f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    return latex


def generate_csv(results: Dict, output_path: str):
    """Generate CSV summary of all results."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Phase", "Config", "Dice", "LVBP", "LVM", "RV", "HD95"])

        for phase_name in sorted(results.keys()):
            phase_results = results[phase_name]
            if isinstance(phase_results, dict):
                for config_name, r in phase_results.items():
                    if isinstance(r, dict) and "mean_dice" in r:
                        writer.writerow([
                            phase_name, config_name,
                            f"{r['mean_dice']:.4f}",
                            f"{r.get('dice_lvbp', 0):.4f}",
                            f"{r.get('dice_lvm', 0):.4f}",
                            f"{r.get('dice_rv', 0):.4f}",
                            f"{r.get('hd95', 0):.2f}",
                        ])


def main():
    parser = argparse.ArgumentParser(description="Generate result tables")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory containing phase results")
    parser.add_argument("--clinical_stats", type=str, default=None,
                        help="Path to clinical statistics.json")
    parser.add_argument("--output_dir", type=str, default="./tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_results(args.results_dir)

    # Generate LaTeX tables
    tables = {
        "table1_phases.tex": generate_table1_phases(results),
        "table2_cross_arch.tex": generate_table2_cross_arch(results),
        "table4_cross_ablation.tex": generate_table4_cross_ablation(results),
    }

    if args.clinical_stats and os.path.exists(args.clinical_stats):
        with open(args.clinical_stats) as f:
            stats = json.load(f)
        tables["table3_clinical.tex"] = generate_table3_clinical(stats)

    for filename, content in tables.items():
        path = os.path.join(args.output_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        print(f"  Generated: {path}")

    # Generate CSV
    csv_path = os.path.join(args.output_dir, "all_results.csv")
    generate_csv(results, csv_path)
    print(f"  Generated: {csv_path}")

    print(f"\nAll tables generated in {args.output_dir}/")


if __name__ == "__main__":
    main()
