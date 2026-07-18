"""
ACDC (Automated Cardiac Diagnosis Challenge) dataset loader.
"""

import os
import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional


class ACDCDataset:
    """
    ACDC dataset loader for cardiac MRI segmentation.

    The ACDC test set contains 50 patients with 100 ED/ES frame pairs.
    Labels: 0=background, 1=LVBP, 2=LVM, 3=RV
    """

    def __init__(self, root_dir: str, split: str = "test"):
        """
        Args:
            root_dir: Path to ACDC dataset root
            split: 'train' or 'test'
        """
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        # Find all patient directories
        self.patient_dirs = sorted([
            d for d in os.listdir(self.split_dir)
            if d.startswith("patient")
        ])

        self.samples = []
        for patient_dir in self.patient_dirs:
            patient_path = os.path.join(self.split_dir, patient_dir)
            # Find .nii.gz files
            files = sorted([f for f in os.listdir(patient_path) if f.endswith('.nii.gz')])

            for f in files:
                if '_gt' in f:
                    continue
                # Extract frame info from filename
                # ACDC format: patientXXX_frameZZ.nii.gz
                frame = f.split('_')[-1].replace('.nii.gz', '')
                gt_file = f.replace('.nii.gz', '_gt.nii.gz')

                self.samples.append({
                    'image_path': os.path.join(patient_path, f),
                    'label_path': os.path.join(patient_path, gt_file) if os.path.exists(os.path.join(patient_path, gt_file)) else None,
                    'patient_id': patient_dir,
                    'frame': frame,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """
        Get a volume.

        Returns:
            image: (D, H, W) numpy array
            label: (D, H, W) numpy array
            patient_id: string
            frame: string ('ED' or 'ES')
        """
        sample = self.samples[idx]

        # Load image
        img_sitk = sitk.ReadImage(sample['image_path'])
        image = sitk.GetArrayFromImage(img_sitk).astype(np.float32)

        # Normalize intensity
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Load label
        if sample['label_path']:
            label_sitk = sitk.ReadImage(sample['label_path'])
            label = sitk.GetArrayFromImage(label_sitk).astype(np.uint8)
        else:
            label = np.zeros_like(image, dtype=np.uint8)

        return image, label, sample['patient_id'], sample['frame']

    def get_spacing(self, idx: int) -> Tuple[float, float, float]:
        """Get the pixel spacing for a volume."""
        sample = self.samples[idx]
        img_sitk = sitk.ReadImage(sample['image_path'])
        spacing = img_sitk.GetSpacing()
        return spacing


def load_acdc_volume(filepath: str) -> np.ndarray:
    """Load a single volume from .nii.gz file."""
    img_sitk = sitk.ReadImage(filepath)
    return sitk.GetArrayFromImage(img_sitk).astype(np.float32)
