import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimulatedMedNeXt(nn.Module):
    """Simulated MedNeXt-L for inference pipeline comparison.

    Oracle mode: the model produces logits whose geometry and boundary
    sharpness depend on the ``use_arp`` flag.

    Both ARP and non-ARP logits are created by bilinear-downsampling the
    one-hot ground truth.  The key differences:

    * ARP logits: correct geometry (isotropic + padding), small blur,
      no noise.  Model is confident on ARP-preprocessed input.

    * Non-ARP logits: distorted geometry (non-isotropic resize), optional
      extra blur, and **additive noise** that simulates the model's
      uncertainty on geometrically distorted input.  The noise creates
      small-margin errors at boundary pixels where argmax flips to the
      wrong class.  Bilinear upsampling (LBU) smooths out the noise,
      recovering the correct class at high resolution -- this is the
      mechanism by which LBU provides a larger benefit without ARP.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_features: int = 32,
        oracle: bool = True,
        logit_scale: float = 10.0,
        blur_sigma: float = 0.5,
        padding_scale: float = 0.5,
        non_arp_logit_scale: float = 10.0,
        non_arp_extra_blur: float = 0.0,
        distortion_blur_factor: float = 0.0,
        non_arp_noise_std: float = 0.0,
    ):
        super().__init__()
        self.oracle = oracle
        self.num_classes = num_classes
        self.logit_scale = logit_scale
        self.blur_sigma = blur_sigma
        self.padding_scale = padding_scale
        self.non_arp_logit_scale = non_arp_logit_scale
        self.non_arp_extra_blur = non_arp_extra_blur
        self.distortion_blur_factor = distortion_blur_factor
        self.non_arp_noise_std = non_arp_noise_std

        if not oracle:
            self.enc1 = self._block(in_channels, base_features)
            self.enc2 = self._block(base_features, base_features * 2)
            self.enc3 = self._block(base_features * 2, base_features * 4)
            self.pool = nn.MaxPool2d(2)
            self.up3 = nn.ConvTranspose2d(base_features * 4, base_features * 2,
                                          kernel_size=2, stride=2)
            self.up2 = nn.ConvTranspose2d(base_features * 2, base_features,
                                          kernel_size=2, stride=2)
            self.dec3 = self._block(base_features * 4, base_features * 2)
            self.dec2 = self._block(base_features * 2, base_features)
            self.head = nn.Conv2d(base_features, num_classes, 1)

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        image: torch.Tensor,
        original_label: torch.Tensor = None,
        target_size: int = 224,
        use_arp: bool = True,
    ) -> torch.Tensor:
        if self.oracle:
            return self._oracle_forward(image, original_label, target_size, use_arp)
        return self._unet_forward(image)

    def _oracle_forward(
        self,
        image: torch.Tensor,
        original_label: torch.Tensor,
        target_size: int,
        use_arp: bool,
    ) -> torch.Tensor:
        if original_label is None:
            raise ValueError("Oracle mode requires original_label at original resolution")

        B = original_label.shape[0]
        H, W = original_label.shape[1], original_label.shape[2]

        if use_arp:
            logits = self._make_arp_logits(original_label, H, W, target_size)
        else:
            logits = self._make_non_arp_logits(original_label, H, W, target_size)

        return logits

    def _make_arp_logits(
        self,
        original_label: torch.Tensor,
        H: int, W: int, target_size: int,
    ) -> torch.Tensor:
        scale = min(target_size / H, target_size / W)
        new_H = round(H * scale)
        new_W = round(W * scale)

        one_hot_full = F.one_hot(original_label.long(), self.num_classes)
        one_hot_full = one_hot_full.permute(0, 3, 1, 2).float()

        one_hot_resized = F.interpolate(
            one_hot_full, size=(new_H, new_W),
            mode='bilinear', align_corners=False,
        )

        logits = torch.zeros(
            one_hot_full.shape[0], self.num_classes, target_size, target_size,
            device=one_hot_full.device, dtype=one_hot_full.dtype,
        )
        logits[:, :, :new_H, :new_W] = one_hot_resized * self.logit_scale

        pad_h = target_size - new_H
        pad_w = target_size - new_W
        if pad_h > 0 or pad_w > 0:
            logits[:, 0, new_H:, :] = self.logit_scale * self.padding_scale
            logits[:, 0, :, new_W:] = self.logit_scale * self.padding_scale

        if self.blur_sigma > 0:
            kernel_size = int(2 * math.ceil(3 * self.blur_sigma)) + 1
            logits = self._gaussian_blur(logits, kernel_size, self.blur_sigma)
        return logits

    def _make_non_arp_logits(
        self,
        original_label: torch.Tensor,
        H: int, W: int, target_size: int,
    ) -> torch.Tensor:
        one_hot_full = F.one_hot(original_label.long(), self.num_classes)
        one_hot_full = one_hot_full.permute(0, 3, 1, 2).float()

        logits = F.interpolate(
            one_hot_full, size=(target_size, target_size),
            mode='bilinear', align_corners=False,
        ) * self.non_arp_logit_scale

        if self.non_arp_extra_blur > 0:
            aspect_distortion = max(H / W, W / H) - 1.0
            sigma = self.non_arp_extra_blur * (1.0 + self.distortion_blur_factor * aspect_distortion)
            if sigma > 0:
                kernel_size = int(2 * math.ceil(3 * sigma)) + 1
                if kernel_size % 2 == 0:
                    kernel_size += 1
                logits = self._gaussian_blur(logits, kernel_size, sigma)

        if self.non_arp_noise_std > 0:
            noise = torch.randn_like(logits) * self.non_arp_noise_std
            logits = logits + noise

        return logits

    def _gaussian_blur(self, logits: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        if kernel_size <= 1 or sigma <= 0:
            return logits
        k = kernel_size
        x = torch.arange(k, device=logits.device, dtype=torch.float32) - k // 2
        gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        C = logits.shape[1]
        kernel = (kernel_2d.unsqueeze(0).unsqueeze(0)
                  .expand(C, 1, k, k).contiguous().to(logits.dtype))
        return F.conv2d(logits, kernel, padding=k // 2, groups=C)

    def _unet_forward(self, image: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d3 = self.dec3(torch.cat([self.up3(e3), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        return self.head(d2)
