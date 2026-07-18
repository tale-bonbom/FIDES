import torch
import torch.nn.functional as F
from .base import BasePipeline


class LBUOnlyPipeline(BasePipeline):
    """LBU-only pipeline: smooth boundaries, geometric distortion.

    Non-ARP preprocessing (non-isotropic resize) + LBU postprocessing
    (bilinear upsample logits to original resolution before argmax).
    The smooth interpolation partially compensates for boundary errors
    caused by the model's confusion on distorted inputs.
    """

    def run(self, image, label, model, target_size):
        B, _, H, W = image.shape

        image_resized = F.interpolate(
            image, size=(target_size, target_size),
            mode='bilinear', align_corners=False,
        )

        logits = model(image_resized, original_label=label,
                       target_size=target_size, use_arp=False)

        logits_upsampled = F.interpolate(
            logits, size=(H, W),
            mode='bilinear', align_corners=False,
        )

        pred = logits_upsampled.argmax(dim=1)

        return pred
