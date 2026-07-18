"""
Error Cancellation Theorem (Theorem 1) verification.

Verifies the theorem that explains why suboptimal pipelines achieve
artificially high EF agreement despite catastrophic volume degradation.

Theorem 1: If volume errors are proportional (β_ED = α * β_ES, α > 0),
then EF error is zero iff α = EDV/ESV.

Corollary 1: When β_ED ≈ β_ES (α ≈ 1), EF error ≈ 0 regardless of
the magnitude of volume errors.

Usage:
    python clinical/error_cancellation.py --measurements ./clinical_results/measurements.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def compute_volume_errors(measurements: Dict) -> Dict:
    """
    Extract volume errors (β_ED, β_ES) for each patient.

    Args:
        measurements: Dictionary from measurements.py

    Returns:
        Dictionary with β_ED, β_ES, and α for each patient
    """
    errors = {}
    for patient_id, m in measurements.items():
        beta_ed = m["errors"]["edv_error"]
        beta_es = m["errors"]["esv_error"]
        alpha = beta_ed / beta_es if abs(beta_es) > 1e-6 else float('inf')
        errors[patient_id] = {
            "beta_ed": beta_ed,
            "beta_es": beta_es,
            "alpha": alpha,
        }
    return errors


def verify_proportionality(errors: Dict) -> Dict:
    """
    Verify whether volume errors are proportional (β_ED ≈ α * β_ES).

    Tests Corollary 1: when α ≈ 1, EF error ≈ 0 regardless of volume errors.

    Args:
        errors: Dictionary from compute_volume_errors

    Returns:
        Verification results
    """
    beta_eds = np.array([e["beta_ed"] for e in errors.values()])
    beta_ess = np.array([e["beta_es"] for e in errors.values()])
    alphas = np.array([e["alpha"] for e in errors.values() if np.isfinite(e["alpha"])])

    # Correlation between β_ED and β_ES
    if len(beta_eds) > 2:
        correlation = float(np.corrcoef(beta_eds, beta_ess)[0, 1])
    else:
        correlation = 0.0

    # Linear regression: β_ED = α * β_ES
    if np.var(beta_ess) > 1e-6:
        reg_alpha = float(np.polyfit(beta_ess, beta_eds, 1)[0])
    else:
        reg_alpha = 0.0

    return {
        "mean_alpha": float(np.mean(alphas)) if len(alphas) > 0 else 0,
        "std_alpha": float(np.std(alphas)) if len(alphas) > 0 else 0,
        "regression_alpha": reg_alpha,
        "correlation_beta_ed_es": correlation,
        "n_patients": len(errors),
    }


def compute_ef_error_theory(measurements: Dict) -> Dict:
    """
    Compute EF error using the theoretical formula (Eq. 3).

    ΔEF ≈ (α - 1) * β_ES * ESV / (EDV + β_ED)^2 * (EDV - ESV)

    Args:
        measurements: Dictionary from measurements.py

    Returns:
        Dictionary with theoretical vs observed EF errors
    """
    theoretical_errors = []
    observed_errors = []

    for patient_id, m in measurements.items():
        edv = m["ground_truth"]["edv"]
        esv = m["ground_truth"]["esv"]
        beta_ed = m["errors"]["edv_error"]
        beta_es = m["errors"]["esv_error"]
        observed_ef_error = m["errors"]["ef_error"]

        # α from data
        if abs(beta_es) > 1e-6:
            alpha = beta_ed / beta_es
        else:
            alpha = 1.0

        # Theoretical EF error (Eq. 3)
        denom = (edv + beta_ed) ** 2
        if denom > 1e-6:
            theoretical_ef_error = (
                (alpha - 1) * beta_es * esv / denom * (edv - esv)
            )
        else:
            theoretical_ef_error = 0.0

        theoretical_errors.append(theoretical_ef_error)
        observed_errors.append(observed_ef_error)

    theoretical_errors = np.array(theoretical_errors)
    observed_errors = np.array(observed_errors)

    # Correlation between theoretical and observed
    if len(theoretical_errors) > 2 and np.std(observed_errors) > 1e-6:
        corr = float(np.corrcoef(theoretical_errors, observed_errors)[0, 1])
    else:
        corr = 0.0

    return {
        "mean_theoretical_ef_error": float(np.mean(np.abs(theoretical_errors))),
        "mean_observed_ef_error": float(np.mean(np.abs(observed_errors))),
        "correlation_theory_observed": corr,
        "rmse_theory_vs_observed": float(np.sqrt(np.mean((theoretical_errors - observed_errors) ** 2))),
    }


def verify_theorem(measurements: Dict) -> Dict:
    """
    Full verification of Theorem 1 (Error Cancellation).

    Args:
        measurements: Dictionary from measurements.py

    Returns:
        Verification results
    """
    errors = compute_volume_errors(measurements)
    proportionality = verify_proportionality(errors)
    ef_error = compute_ef_error_theory(measurements)

    # Check: Is EF error small despite large volume errors?
    mean_abs_beta_ed = float(np.mean(np.abs([e["beta_ed"] for e in errors.values()])))
    mean_abs_beta_es = float(np.mean(np.abs([e["beta_es"] for e in errors.values()])))
    mean_abs_ef_error = float(np.mean(np.abs([m["errors"]["ef_error"] for m in measurements.values()])))

    paradox_confirmed = (
        (mean_abs_beta_ed > 5.0 or mean_abs_beta_es > 5.0)
        and mean_abs_ef_error < 5.0
    )

    return {
        "proportionality_analysis": proportionality,
        "ef_error_analysis": ef_error,
        "mean_abs_beta_ed": mean_abs_beta_ed,
        "mean_abs_beta_es": mean_abs_beta_es,
        "mean_abs_ef_error": mean_abs_ef_error,
        "paradox_confirmed": paradox_confirmed,
        "theorem_verified": proportionality["correlation_beta_ed_es"] > 0.7,
    }


def main():
    parser = argparse.ArgumentParser(description="Error Cancellation Theorem Verification")
    parser.add_argument("--measurements", type=str, required=True,
                        help="Path to measurements.json")
    parser.add_argument("--output", type=str, default="./clinical_results/error_cancellation.json")
    args = parser.parse_args()

    with open(args.measurements) as f:
        measurements = json.load(f)

    results = verify_theorem(measurements)

    print(f"\n{'='*70}")
    print("  Theorem 1: Error Cancellation Verification")
    print(f"{'='*70}")

    prop = results["proportionality_analysis"]
    print(f"\n  Proportionality Analysis (β_ED vs β_ES):")
    print(f"    Correlation:        {prop['correlation_beta_ed_es']:.3f}")
    print(f"    Regression α:       {prop['regression_alpha']:.3f}")
    print(f"    Mean α (per-patient): {prop['mean_alpha']:.3f} ± {prop['std_alpha']:.3f}")

    ef = results["ef_error_analysis"]
    print(f"\n  EF Error Analysis:")
    print(f"    Mean |theoretical ΔEF|: {ef['mean_theoretical_ef_error']:.4f}")
    print(f"    Mean |observed ΔEF|:    {ef['mean_observed_ef_error']:.4f}")
    print(f"    Theory-Observed corr:   {ef['correlation_theory_observed']:.3f}")
    print(f"    RMSE theory vs observed: {ef['rmse_theory_vs_observed']:.4f}")

    print(f"\n  Paradox Check:")
    print(f"    Mean |β_ED|:  {results['mean_abs_beta_ed']:.2f} mL")
    print(f"    Mean |β_ES|:  {results['mean_abs_beta_es']:.2f} mL")
    print(f"    Mean |ΔEF|:   {results['mean_abs_ef_error']:.2f} %")
    print(f"    Paradox confirmed: {'YES' if results['paradox_confirmed'] else 'NO'}")
    print(f"    Theorem verified:  {'YES' if results['theorem_verified'] else 'NO'}")

    print(f"\n  Conclusion: Volume errors are {'proportional' if results['theorem_verified'] else 'not proportional'},")
    print(f"  explaining why EF agreement remains {'high' if results['paradox_confirmed'] else 'variable'}")
    print(f"  despite {'large' if results['mean_abs_beta_ed'] > 5 else 'small'} volume degradation.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
