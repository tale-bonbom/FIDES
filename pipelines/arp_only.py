import torch
import torch.nn.functional as F
from .base import BasePipeline


class ARPOnlyPipeline(BasePipeline):
    """ARP-only pipeline: correct geometry, mosaic boundaries.

    ARP preprocessing (isotropic resize + zero-padding) + non-LBU postprocessing
    (crop content region, argmax at low resolution, nearest-neighbour resize).
    Preserves correct aspect ratio but boundaries remain blocky (mosaic).
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

        pred_low = logits.argmax(dim=1)
        pred_cropped = pred_low[:, :new_H, :new_W]

        pred = F.interpolate(
            pred_cropped.unsqueeze(1).float(), size=(H, W),
            mode='nearest',
        ).squeeze(1).long()

        return pred
