"""
Statistical analysis: ICC, Bland-Altman, Wilcoxon tests.

Provides intraclass correlation coefficient (ICC(2,1)), Bland-Altman
analysis, and Wilcoxon signed-rank tests for clinical metric validation.

Usage:
    python clinical/statistics.py --measurements ./clinical_results/measurements.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats


def icc_2_1(ratings: np.ndarray) -> Dict:
    """
    Compute ICC(2,1) — two-way random, single measures.

    ICC(2,1) is the appropriate model when:
    - Both raters (model and ground truth) are considered random
    - Single measurements are used
    - Used for absolute agreement

    Args:
        ratings: 2D array of shape (n_subjects, 2) where column 0 is
                 the model prediction and column 1 is the ground truth.

    Returns:
        Dictionary with ICC value and confidence interval
    """
    n, k = ratings.shape  # n subjects, k raters
    if k != 2:
        raise ValueError("ICC(2,1) requires exactly 2 raters.")

    grand_mean = ratings.mean()
    ss_between = n * np.var(ratings.mean(axis=1), ddof=1)
    ss_within = np.var(ratings, axis=1, ddof=1).sum() * (k - 1) / k
    ss_rater = n * np.var(ratings.mean(axis=0), ddof=1)
    ss_error = ss_within - ss_rater

    ms_between = ss_between / (n - 1)
    ms_rater = ss_rater / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    # ICC(2,1) formula
    icc = (ms_between - ms_error) / (ms_between + (k - 1) * ms_error + k * (ms_rater - ms_error) / n)

    # F-statistic for significance
    f_stat = ms_between / ms_error if ms_error > 0 else float('inf')
    p_value = 1 - stats.f.cdf(f_stat, n - 1, (n - 1) * (k - 1)) if ms_error > 0 else 0.0

    # 95% CI
    fl = (f_stat / stats.f.ppf(0.975, n - 1, (n - 1) * (k - 1)))
    fu = (f_stat / stats.f.ppf(0.025, n - 1, (n - 1) * (k - 1)))
    icc_lower = (fl - 1) / (fl + (k - 1) + k * (ms_rater - ms_error) / (n * ms_error)) if ms_error > 0 else 0
    icc_upper = (fu - 1) / (fu + (k - 1) + k * (ms_rater - ms_error) / (n * ms_error)) if ms_error > 0 else 1

    return {
        "icc": float(icc),
        "icc_lower": float(max(0, icc_lower)),
        "icc_upper": float(min(1, icc_upper)),
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "n_subjects": int(n),
    }


def bland_altman(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> Dict:
    """
    Compute Bland-Altman analysis for agreement between two measurements.

    Args:
        predicted: Model predictions (n,)
        reference: Ground truth values (n,)

    Returns:
        Dictionary with mean bias, LoA, and percentages
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    diff = predicted - reference
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    mean_val = np.mean((predicted + reference) / 2)

    loa_lower = mean_diff - 1.96 * std_diff
    loa_upper = mean_diff + 1.96 * std_diff

    # Percentage limits of agreement
    pct_bias = (mean_diff / mean_val) * 100 if mean_val != 0 else 0
    pct_loa = (1.96 * std_diff / mean_val) * 100 if mean_val != 0 else 0

    return {
        "mean_bias": float(mean_diff),
        "std_bias": float(std_diff),
        "loa_lower": float(loa_lower),
        "loa_upper": float(loa_upper),
        "pct_bias": float(pct_bias),
        "pct_loa": float(pct_loa),
        "mean_value": float(mean_val),
        "n": int(len(diff)),
    }


def wilcoxon_test(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> Dict:
    """
    Wilcoxon signed-rank test for paired samples.

    Tests whether there is a significant difference between paired
    predicted and reference values (non-parametric).

    Args:
        predicted: Model predictions (n,)
        reference: Ground truth values (n,)

    Returns:
        Dictionary with statistic and p-value
    """
    stat, p_value = stats.wilcoxon(predicted, reference)

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n": int(len(predicted)),
    }


def mean_absolute_error(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(reference))))


def root_mean_squared_error(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(reference)) ** 2)))


def pearson_correlation(predicted: np.ndarray, reference: np.ndarray) -> Dict:
    """Compute Pearson correlation coefficient."""
    r, p = stats.pearsonr(predicted, reference)
    return {"r": float(r), "p_value": float(p)}


def run_full_analysis(measurements: Dict) -> Dict:
    """
    Run full statistical analysis on clinical measurements.

    Args:
        measurements: Dictionary of patient measurements (from measurements.py)

    Returns:
        Dictionary with all statistical results
    """
    edv_pred, edv_gt = [], []
    esv_pred, esv_gt = [], []
    ef_pred, ef_gt = [], []

    for patient_id, m in measurements.items():
        edv_pred.append(m["predicted"]["edv"])
        edv_gt.append(m["ground_truth"]["edv"])
        esv_pred.append(m["predicted"]["esv"])
        esv_gt.append(m["ground_truth"]["esv"])
        ef_pred.append(m["predicted"]["ef"])
        ef_gt.append(m["ground_truth"]["ef"])

    edv_pred, edv_gt = np.array(edv_pred), np.array(edv_gt)
    esv_pred, esv_gt = np.array(esv_pred), np.array(esv_gt)
    ef_pred, ef_gt = np.array(ef_pred), np.array(ef_gt)

    results = {}
    for name, pred, gt in [("EDV", edv_pred, edv_gt), ("ESV", esv_pred, esv_gt), ("EF", ef_pred, ef_gt)]:
        ratings = np.column_stack([pred, gt])
        results[name] = {
            "icc": icc_2_1(ratings),
            "bland_altman": bland_altman(pred, gt),
            "wilcoxon": wilcoxon_test(pred, gt),
            "mae": mean_absolute_error(pred, gt),
            "rmse": root_mean_squared_error(pred, gt),
            "pearson": pearson_correlation(pred, gt),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Statistical Analysis (ICC, Bland-Altman, Wilcoxon)")
    parser.add_argument("--measurements", type=str, required=True,
                        help="Path to measurements.json")
    parser.add_argument("--output", type=str, default="./clinical_results/statistics.json")
    args = parser.parse_args()

    with open(args.measurements) as f:
        measurements = json.load(f)

    results = run_full_analysis(measurements)

    print(f"\n{'='*70}")
    print("  Clinical Statistical Analysis Results")
    print(f"{'='*70}")
    for metric, stats_dict in results.items():
        print(f"\n  {metric}:")
        print(f"    ICC(2,1):  {stats_dict['icc']['icc']:.3f} "
              f"[{stats_dict['icc']['icc_lower']:.3f}, {stats_dict['icc']['icc_upper']:.3f}]")
        print(f"    MAE:       {stats_dict['mae']:.2f}")
        print(f"    RMSE:      {stats_dict['rmse']:.2f}")
        print(f"    Pearson r: {stats_dict['pearson']['r']:.3f}")
        ba = stats_dict["bland_altman"]
        print(f"    Bias:      {ba['mean_bias']:.2f} (LoA: [{ba['loa_lower']:.2f}, {ba['loa_upper']:.2f}])")
        print(f"    Wilcoxon:  p={stats_dict['wilcoxon']['p_value']:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nStatistics saved to {args.output}")


if __name__ == "__main__":
    main()
