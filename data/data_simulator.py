import numpy as np
from typing import List, Tuple, Optional

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


class CardiacMRISimulator:
    """Simulate cardiac MRI short-axis slices with anatomical structures.

    Generates 2D images containing:
        - Class 0: Background
        - Class 1: LVBP (Left Ventricle Blood Pool) - bright ellipse
        - Class 2: LVM  (Left Ventricle Myocardium) - ring around LVBP
        - Class 3: RV   (Right Ventricle) - crescent shape

    When variable_size=True, images have random spatial dimensions to
    simulate the varying aspect ratios found in real cardiac MRI data.
    This makes the geometric distortion from non-ARP resizing clearly visible.
    """

    def __init__(
        self,
        num_classes: int = 4,
        image_size: int = 256,
        variable_size: bool = True,
        size_range: Tuple[int, int] = (100, 400),
        noise_std: float = 0.15,
    ):
        self.num_classes = num_classes
        self.image_size = image_size
        self.variable_size = variable_size
        self.size_range = size_range
        self.noise_std = noise_std

    def generate(self, n_samples: int = 20, seed: Optional[int] = None
                 ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if seed is not None:
            np.random.seed(seed)
        images, labels = [], []
        for _ in range(n_samples):
            if self.variable_size:
                H = np.random.randint(self.size_range[0], self.size_range[1] + 1)
                W = np.random.randint(self.size_range[0], self.size_range[1] + 1)
            else:
                H = W = self.image_size
            img, lbl = self._generate_one(H, W)
            images.append(img)
            labels.append(lbl)
        return images, labels

    def _generate_one(self, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
        image = np.zeros((H, W), dtype=np.float32)
        label = np.zeros((H, W), dtype=np.int64)

        cx = W * np.random.uniform(0.40, 0.50)
        cy = H * np.random.uniform(0.45, 0.55)
        angle = np.random.uniform(-0.2, 0.2)

        Y, X = np.mgrid[:H, :W]
        Xr = (X - cx) * np.cos(angle) + (Y - cy) * np.sin(angle)
        Yr = -(X - cx) * np.sin(angle) + (Y - cy) * np.cos(angle)

        lvbp_rx = W * np.random.uniform(0.12, 0.17)
        lvbp_ry = H * np.random.uniform(0.14, 0.20)
        lvbp_mask = (Xr / lvbp_rx) ** 2 + (Yr / lvbp_ry) ** 2 <= 1
        label[lvbp_mask] = 1
        image[lvbp_mask] = np.random.uniform(0.55, 0.70)

        lvm_rx = lvbp_rx * np.random.uniform(1.06, 1.14)
        lvm_ry = lvbp_ry * np.random.uniform(1.06, 1.14)
        lvm_mask = ((Xr / lvm_rx) ** 2 + (Yr / lvm_ry) ** 2 <= 1) & ~lvbp_mask
        label[lvm_mask] = 2
        image[lvm_mask] = np.random.uniform(0.30, 0.45)

        rv_cx = cx + W * np.random.uniform(0.15, 0.25)
        rv_cy = cy - H * np.random.uniform(0.00, 0.10)
        rv_rx = W * np.random.uniform(0.14, 0.20)
        rv_ry = H * np.random.uniform(0.16, 0.23)
        rv_angle = np.random.uniform(-0.3, 0.3)
        Xr_rv = (X - rv_cx) * np.cos(rv_angle) + (Y - rv_cy) * np.sin(rv_angle)
        Yr_rv = -(X - rv_cx) * np.sin(rv_angle) + (Y - rv_cy) * np.cos(rv_angle)

        rv_outer = (Xr_rv / rv_rx) ** 2 + (Yr_rv / rv_ry) ** 2 <= 1
        rv_inner_rx = rv_rx * np.random.uniform(0.84, 0.94)
        rv_inner_ry = rv_ry * np.random.uniform(0.84, 0.94)
        rv_inner = (Xr_rv / rv_inner_rx) ** 2 + (Yr_rv / rv_inner_ry) ** 2 <= 1
        rv_mask = rv_outer & ~rv_inner & ~lvm_mask & ~lvbp_mask
        label[rv_mask] = 3
        image[rv_mask] = np.random.uniform(0.50, 0.65)

        image += np.random.randn(H, W) * self.noise_std
        image = np.clip(image, 0, 1)
        return image, label

    @staticmethod
    def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray]:
        if not HAS_NIBABEL:
            raise ImportError("nibabel is required for loading NIfTI files. "
                              "Install it with: pip install nibabel")
        nib_img = nib.load(path)
        image = nib_img.get_fdata().astype(np.float32)
        if image.ndim == 3:
            slice_idx = image.shape[2] // 2
            image = image[:, :, slice_idx]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        label = np.zeros(image.shape, dtype=np.int64)
        return image, label
