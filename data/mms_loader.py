"""
M&Ms (Multi-Centre, Multi-Vendor & Multi-Disease) dataset loader.
"""

import os
import numpy as np
import SimpleITK as sitk
from typing import Tuple


class MMsDataset:
    """
    M&Ms dataset loader for cross-domain validation.

    The M&Ms dataset introduces domain shift through multi-center,
    multi-vendor acquisition.
    """

    def __init__(self, root_dir: str, split: str = "test"):
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        self.samples = []
        for root, dirs, files in os.walk(self.split_dir):
            for f in sorted(files):
                if f.endswith('.nii.gz') and '_gt' not in f:
                    gt_file = f.replace('.nii.gz', '_gt.nii.gz')
                    self.samples.append({
                        'image_path': os.path.join(root, f),
                        'label_path': os.path.join(root, gt_file) if os.path.exists(os.path.join(root, gt_file)) else None,
                        'patient_id': f.replace('.nii.gz', ''),
                        'frame': 'unknown',
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
        sample = self.samples[idx]
        img_sitk = sitk.ReadImage(sample['image_path'])
        image = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        if sample['label_path']:
            label_sitk = sitk.ReadImage(sample['label_path'])
            label = sitk.GetArrayFromImage(label_sitk).astype(np.uint8)
        else:
            label = np.zeros_like(image, dtype=np.uint8)

        return image, label, sample['patient_id'], sample['frame']
