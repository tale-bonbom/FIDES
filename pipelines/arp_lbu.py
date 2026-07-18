import torch
import torch.nn.functional as F
from .base import BasePipeline


class ARPLBUPipeline(BasePipeline):
    """FIDES-optimal pipeline: ARP + LBU combined.

    ARP preprocessing (isotropic resize + zero-padding) + LBU postprocessing
    (crop content region, bilinear upsample logits to original resolution
    before argmax).  Both geometric distortion and mosaic boundaries are
    eliminated, yielding the best Dice.
    """

    def run(self, image, label, model, target_size):
        B, _, H, W = image.shape

        scale = min(target_size / H, target_size / W)
        new_H = round(H * scale)
        new_W = round(W * scale)

        image_resized = F.interpolate(
            image, size=(new_H, new_W),
            mode='bilinear', align_corners=False,
        )

        pad_h = target_size - new_H
        pad_w = target_size - new_W
        image_padded = F.pad(image_resized, (0, pad_w, 0, pad_h), value=0)

        logits = model(image_padded, original_label=label,
                       target_size=target_size, use_arp=True)

        logits_cropped = logits[:, :, :new_H, :new_W]

        logits_upsampled = F.interpolate(
            logits_cropped, size=(H, W),
            mode='bilinear', align_corners=False,
        )

        pred = logits_upsampled.argmax(dim=1)

        return pred
