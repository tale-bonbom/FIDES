import torch
import torch.nn.functional as F
from .base import BasePipeline


class DefectivePipeline(BasePipeline):
    """Defective inference pipeline (CorSeg baseline).

    Non-ARP preprocessing (non-isotropic resize) + non-LBU postprocessing
    (argmax at low resolution then nearest-neighbour resize).
    Produces both geometric distortion and mosaic (staircase) boundaries.
    """

    def run(self, image, label, model, target_size):
        B, _, H, W = image.shape

        image_resized = F.interpolate(
            image, size=(target_size, target_size),
            mode='bilinear', align_corners=False,
        )

        logits = model(image_resized, original_label=label,
                       target_size=target_size, use_arp=False)

        pred_low = logits.argmax(dim=1)

        pred = F.interpolate(
            pred_low.unsqueeze(1).float(), size=(H, W),
            mode='nearest',
        ).squeeze(1).long()

        return pred
