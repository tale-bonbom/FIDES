"""
Clinical measurements: EDV, ESV, EF calculation.

Computes ventricular volumes and ejection fraction from 3D segmentation
masks. Labels: 0=background, 1=LVBP (left ventricle blood pool),
2=LVM (left ventricle myocardium), 3=RV (right ventricle).

Usage:
    python clinical/measurements.py --predictions ./results --acdc_dir /path/to/acdc
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import SimpleITK as sitk
from typing import Dict, Tuple


# Label constants
LVBP = 1  # Left ventricle blood pool
LVM = 2  # Left ventricle myocardium
RV = 3   # Right ventricle


def compute_volume_ml(
    segmentation: np.ndarray,
    label: int,
    pixel_spacing: Tuple[float, float, float],
    slice_thickness: float,
) -> float:
    """
    Compute volume in milliliters (mL) for a given structure label.

    Args:
        segmentation: 3D segmentation mask (D, H, W)
        label: Structure label (1=LVBP, 2=LVM, 3=RV)
        pixel_spacing: In-plane spacing (x, y) in mm
        slice_thickness: Slice thickness in mm

    Returns:
        Volume in mL (1 mL = 1000 mm^3)
    """
    voxel_count = int((segmentation == label).sum())
    voxel_volume_mm3 = (
        pixel_spacing[0] * pixel_spacing[1] * slice_thickness
    )
    volume_ml = voxel_count * voxel_volume_mm3 / 1000.0
    return volume_ml


def compute_edv(
    segmentation_ed: np.ndarray,
    pixel_spacing: Tuple[float, float, float],
    slice_thickness: float,
) -> float:
    """
    Compute End-Diastolic Volume (EDV) from the ED frame segmentation.

    EDV = volume of LV blood pool at end-diastole.

    Args:
        segmentation_ed: ED frame segmentation (D, H, W)
        pixel_spacing: In-plane spacing (x, y) in mm
        slice_thickness: Slice thickness in mm

    Returns:
        EDV in mL
    """
    return compute_volume_ml(segmentation_ed, LVBP, pixel_spacing, slice_thickness)


def compute_esv(
    segmentation_es: np.ndarray,
    pixel_spacing: Tuple[float, float, float],
    slice_thickness: float,
) -> float:
    """
    Compute End-Systolic Volume (ESV) from the ES frame segmentation.

    ESV = volume of LV blood pool at end-systole.

    Args:
        segmentation_es: ES frame segmentation (D, H, W)
        pixel_spacing: In-plane spacing (x, y) in mm
        slice_thickness: Slice thickness in mm

    Returns:
        ESV in mL
    """
    return compute_volume_ml(segmentation_es, LVBP, pixel_spacing, slice_thickness)


def compute_ef(edv: float, esv: float) -> float:
    """
    Compute Ejection Fraction (EF).

    EF = (EDV - ESV) / EDV * 100 (%)

    Args:
        edv: End-diastolic volume in mL
        esv: End-systolic volume in mL

    Returns:
        EF as percentage (0-100)
    """
    if edv < 1e-6:
        return 0.0
    return (edv - esv) / edv * 100.0


def compute_myocardial_mass(
    segmentation_ed: np.ndarray,
    pixel_spacing: Tuple[float, float, float],
    slice_thickness: float,
    density: float = 1.05,
) -> float:
    """
    Compute Left Ventricular Myocardial Mass (LVM mass) at ED.

    Args:
        segmentation_ed: ED frame segmentation (D, H, W)
        pixel_spacing: In-plane spacing (x, y) in mm
        slice_thickness: Slice thickness in mm
        density: Myocardial tissue density (g/mL), default 1.05

    Returns:
        LVM mass in grams
    """
    volume_ml = compute_volume_ml(segmentation_ed, LVM, pixel_spacing, slice_thickness)
    return volume_ml * density


def compute_stroke_volume(edv: float, esv: float) -> float:
    """Compute Stroke Volume (SV) = EDV - ESV."""
    return edv - esv


def compute_cardiac_output(sv: float, heart_rate: int = 60) -> float:
    """Compute Cardiac Output (CO) = SV * HR / 1000 (L/min)."""
    return sv * heart_rate / 1000.0


def compute_clinical_metrics(
    prediction_ed: np.ndarray,
    prediction_es: np.ndarray,
    gt_ed: np.ndarray,
    gt_es: np.ndarray,
    pixel_spacing: Tuple[float, float, float],
    slice_thickness: float,
) -> Dict:
    """
    Compute all clinical metrics for a patient.

    Args:
        prediction_ed: Predicted ED segmentation (D, H, W)
        prediction_es: Predicted ES segmentation (D, H, W)
        gt_ed: Ground truth ED segmentation (D, H, W)
        gt_es: Ground truth ES segmentation (D, H, W)
        pixel_spacing: In-plane spacing (x, y) in mm
        slice_thickness: Slice thickness in mm

    Returns:
        Dictionary with predicted and GT clinical metrics
    """
    # Predicted
    pred_edv = compute_edv(prediction_ed, pixel_spacing, slice_thickness)
    pred_esv = compute_esv(prediction_es, pixel_spacing, slice_thickness)
    pred_ef = compute_ef(pred_edv, pred_esv)
    pred_sv = compute_stroke_volume(pred_edv, pred_esv)
    pred_mass = compute_myocardial_mass(prediction_ed, pixel_spacing, slice_thickness)

    # Ground truth
    gt_edv = compute_edv(gt_ed, pixel_spacing, slice_thickness)
    gt_esv = compute_esv(gt_es, pixel_spacing, slice_thickness)
    gt_ef = compute_ef(gt_edv, gt_esv)
    gt_sv = compute_stroke_volume(gt_edv, gt_esv)
    gt_mass = compute_myocardial_mass(gt_ed, pixel_spacing, slice_thickness)

    return {
        "predicted": {
            "edv": pred_edv,
            "esv": pred_esv,
            "ef": pred_ef,
            "sv": pred_sv,
            "lvm_mass": pred_mass,
        },
        "ground_truth": {
            "edv": gt_edv,
            "esv": gt_esv,
            "ef": gt_ef,
            "sv": gt_sv,
            "lvm_mass": gt_mass,
        },
        "errors": {
            "edv_error": pred_edv - gt_edv,
            "esv_error": pred_esv - gt_esv,
            "ef_error": pred_ef - gt_ef,
            "sv_error": pred_sv - gt_sv,
            "mass_error": pred_mass - gt_mass,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Clinical Measurements (EDV, ESV, EF)")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to prediction directory")
    parser.add_argument("--acdc_dir", type=str, required=True,
                        help="Path to ACDC dataset (for ground truth)")
    parser.add_argument("--output", type=str, default="./clinical_results/measurements.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load predictions and ground truth
    from data.acdc_loader import ACDCDataset
    dataset = ACDCDataset(args.acdc_dir, split="test")

    all_measurements = {}
    for idx in range(len(dataset)):
        image, gt_label, patient_id, frame = dataset[idx]
        spacing = dataset.get_spacing(idx)
        pixel_spacing = (spacing[0], spacing[1])
        slice_thickness = spacing[2] if len(spacing) > 2 else 8.0

        if frame == "ED":
            pred_path = os.path.join(args.predictions, f"{patient_id}_ED_pred.nii.gz")
            if not os.path.exists(pred_path):
                continue
            pred_ed = sitk.GetArrayFromImage(sitk.ReadImage(pred_path)).astype(np.uint8)

            # Find matching ES frame
            for idx2 in range(len(dataset)):
                _, gt_es, pid2, frame2 = dataset[idx2]
                if pid2 == patient_id and frame2 == "ES":
                    pred_path_es = os.path.join(args.predictions, f"{patient_id}_ES_pred.nii.gz")
                    if not os.path.exists(pred_path_es):
                        break
                    pred_es = sitk.GetArrayFromImage(sitk.ReadFile(pred_path_es)).astype(np.uint8)

                    metrics = compute_clinical_metrics(
                        pred_ed, pred_es, gt_label, gt_es,
                        pixel_spacing, slice_thickness,
                    )
                    all_measurements[patient_id] = metrics
                    print(f"  {patient_id}: pred EF={metrics['predicted']['ef']:.1f}%, "
                          f"GT EF={metrics['ground_truth']['ef']:.1f}%")
                    break

    with open(args.output, "w") as f:
        json.dump(all_measurements, f, indent=2)
    print(f"\nMeasurements saved to {args.output}")


if __name__ == "__main__":
    main()
