"""
FIDES pipeline implementation.

This is the core pipeline that orchestrates ARP, PSR, LBU, TTA, and
post-processing according to a PipelineConfig. It implements the
Training-Inference Consistency Guideline (TICP).
"""

import os
import time
import torch
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

from fides.configs import PipelineConfig
from fides.transforms import (
    aspect_ratio_preservation,
    pixel_spacing_restoration,
    logits_bilinear_upsampling,
    apply_normalization,
)
from fides.tta import apply_tta, apply_continuous_rotation_tta
from fides.postprocessing import apply_postprocessing
from fides.model import MedNeXtL
from utils.metrics import dice_score, hausdorff_distance_95


class FIDESPipeline:
    """
    FIDES inference pipeline.

    Implements the Training-Inference Consistency Guideline (TICP) through
    four configurable components:
    1. Aspect Ratio Preservation (ARP)
    2. Pixel Spacing Restoration (PSR)
    3. Logits-Bilinear-Upsampling (LBU)
    4. Extended TTA (eTTA-HVR)
    """

    def __init__(
        self,
        config: PipelineConfig,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        input_size: int = 224,
        num_classes: int = 4,
    ):
        self.config = config
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes

        # Initialize model
        self.model = MedNeXtL(
            in_channels=1,
            num_classes=num_classes,
            input_size=input_size,
            kernel_size=5,
        )

        if checkpoint_path:
            self.model.load_checkpoint(checkpoint_path)

        self.model = self.model.to(device)
        self.model.eval()

    def preprocess(self, image: np.ndarray) -> tuple:
        """
        Preprocess a single slice.

        Steps:
        1. (Optional) Aspect Ratio Preservation (ARP)
        2. Z-score normalization
        3. Resize to input size
        """
        # ARP: pad to square before resize
        if self.config.arp:
            image_proc, metadata = aspect_ratio_preservation(
                image, target_size=self.input_size
            )
        else:
            # CorSeg-Original: non-isotropic resize (BROKEN)
            import torch.nn.functional as F
            if image.ndim == 2:
                img_t = torch.from_numpy(image[None, None].astype(np.float32))
            else:
                img_t = torch.from_numpy(image[None].transpose(2, 0, 1).astype(np.float32))
            resized = F.interpolate(img_t, size=(self.input_size, self.input_size),
                                    mode='bilinear', align_corners=False)
            image_proc = resized.numpy()[0, 0] if image.ndim == 2 else resized.numpy()[0].transpose(1, 2, 0)
            metadata = {
                'orig_h': image.shape[0], 'orig_w': image.shape[1],
                'pad_h': 0, 'pad_w': 0, 'pad_h_extra': 0, 'pad_w_extra': 0,
                'padded_size': self.input_size, 'target_size': self.input_size,
            }

        # Normalization
        norm_method = self.config.normalization if self.config.normalization else "zscore"
        image_proc = apply_normalization(image_proc, method=norm_method)

        # Convert to tensor
        image_tensor = torch.from_numpy(image_proc.astype(np.float32))
        image_tensor = image_tensor[None, None] if image_tensor.ndim == 2 else image_tensor[None]
        image_tensor = image_tensor.to(self.device)

        return image_tensor, metadata

    def inference(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Run model inference with optional TTA."""
        if self.config.tta:
            # Discrete TTA (flips, 90° rotations) - TICP consistent
            softmax = apply_tta(self.model, image_tensor, self.config.tta)
        elif self.config.continuous_rotation is not None:
            # Continuous rotation TTA - TICP inconsistent (for ablation)
            angles = [-self.config.continuous_rotation, 0, self.config.continuous_rotation]
            softmax = apply_continuous_rotation_tta(self.model, image_tensor, angles)
        else:
            # No TTA
            with torch.no_grad():
                logits = self.model(image_tensor)
            softmax = torch.softmax(logits, dim=1)

        return softmax

    def postprocess(
        self,
        softmax: torch.Tensor,
        metadata: dict,
    ) -> np.ndarray:
        """
        Postprocess model output.

        Steps:
        1. (Optional) LBU: upsample logits before argmax
        2. (Optional) PSR: restore original pixel spacing
        3. Argmax to get prediction
        4. (Optional) Post-processing
        """
        orig_h = metadata['orig_h']
        orig_w = metadata['orig_w']

        softmax = softmax.cpu()

        # LBU: upsample continuous logits
        if self.config.lbu:
            # First resize back to padded size
            padded_size = metadata['padded_size']
            if softmax.ndim == 4:
                softmax = F.interpolate(
                    softmax, size=(padded_size, padded_size),
                    mode='bilinear', align_corners=False
                )
            else:
                softmax = F.interpolate(
                    softmax[None], size=(padded_size, padded_size),
                    mode='bilinear', align_corners=False
                )[0]

            # PSR: remove padding
            pad_h = metadata.get('pad_h', 0)
            pad_w = metadata.get('pad_w', 0)
            pad_h_extra = metadata.get('pad_h_extra', 0)
            pad_w_extra = metadata.get('pad_w_extra', 0)

            if pad_h > 0 or pad_h_extra > 0:
                if softmax.ndim == 4:
                    softmax = softmax[:, :, pad_h:pad_h + orig_h, :]
                else:
                    softmax = softmax[:, pad_h:pad_h + orig_h, :]

            if pad_w > 0 or pad_w_extra > 0:
                if softmax.ndim == 4:
                    softmax = softmax[:, :, :, pad_w:pad_w + orig_w]
                else:
                    softmax = softmax[:, :, pad_w:pad_w + orig_w]
        else:
            # CorSeg-Original: argmax before upsampling (BROKEN - mosaic effect)
            import torch.nn.functional as F
            pred = torch.argmax(softmax, dim=1 if softmax.ndim == 4 else 0)
            if pred.ndim == 3:
                pred = pred[0]
            # Nearest-neighbor upsampling of discrete labels
            pred_4d = pred[None, None].float()
            upsampled = F.interpolate(pred_4d, size=(orig_h, orig_w), mode='nearest')
            pred_np = upsampled[0, 0].numpy().astype(np.uint8)

            # Apply post-processing if specified
            if self.config.postprocessing:
                pred_np = apply_postprocessing(pred_np, self.config.postprocessing)

            return pred_np

        # Argmax on continuous logits
        if softmax.ndim == 4:
            pred = torch.argmax(softmax[0], dim=0).numpy().astype(np.uint8)
        else:
            pred = torch.argmax(softmax, dim=0).numpy().astype(np.uint8)

        # Apply post-processing if specified
        if self.config.postprocessing:
            pred = apply_postprocessing(pred, self.config.postprocessing)

        return pred

    def predict_slice(self, image: np.ndarray) -> np.ndarray:
        """Run full pipeline on a single 2D slice."""
        # Preprocess
        image_tensor, metadata = self.preprocess(image)

        # Inference
        softmax = self.inference(image_tensor)

        # Postprocess
        pred = self.postprocess(softmax, metadata)

        return pred

    def run(self, dataset, output_dir: str) -> Dict:
        """
        Run the pipeline on a dataset.

        Args:
            dataset: Dataset with __getitem__ returning (image, label, patient_id, frame)
            output_dir: Directory to save predictions

        Returns:
            Dictionary of results
        """
        os.makedirs(output_dir, exist_ok=True)

        all_dice = []
        all_hd95 = []
        per_structure_dice = {'lvbp': [], 'lvm': [], 'rv': []}

        for i in tqdm(range(len(dataset)), desc=f"Running {self.config.name}"):
            image, label, patient_id, frame = dataset[i]

            # Predict each slice
            pred_volume = []
            for s in range(image.shape[0]):
                pred_slice = self.predict_slice(image[s])
                pred_volume.append(pred_slice)
            pred_volume = np.stack(pred_volume)

            # Save prediction
            np.save(os.path.join(output_dir, f"{patient_id}_{frame}_pred.npy"), pred_volume)

            # Compute metrics
            dice = dice_score(pred_volume, label)
            hd95 = hausdorff_distance_95(pred_volume, label)

            all_dice.append(dice['mean'])
            all_hd95.append(hd95)
            per_structure_dice['lvbp'].append(dice['lvbp'])
            per_structure_dice['lvm'].append(dice['lvm'])
            per_structure_dice['rv'].append(dice['rv'])

        results = {
            'mean_dice': float(np.mean(all_dice)),
            'std_dice': float(np.std(all_dice)),
            'dice_lvbp': float(np.mean(per_structure_dice['lvbp'])),
            'dice_lvm': float(np.mean(per_structure_dice['lvm'])),
            'dice_rv': float(np.mean(per_structure_dice['rv'])),
            'hd95': float(np.mean(all_hd95)),
            'all_dice': all_dice,
        }

        return results
