"""
MedNeXt-L model wrapper for cardiac MRI segmentation.

MedNeXt: Transformer-driven scaling of ConvNets for medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MedNeXtBlock(nn.Module):
    """MedNeXt block with depthwise separable convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv3d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels
        )
        self.norm = nn.InstanceNorm3d(in_channels)
        self.act = nn.GELU()
        self.pwconv1 = nn.Conv3d(in_channels, out_channels * 4, 1)
        self.pwconv2 = nn.Conv3d(out_channels * 4, out_channels, 1)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + residual


class MedNeXtEncoder(nn.Module):
    """Simplified MedNeXt encoder for 2D slice processing."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_classes: int = 4,
        kernel_size: int = 5,
        input_size: int = 224,
    ):
        super().__init__()
        self.input_size = input_size

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.GELU(),
        )

        # Encoder stages
        self.enc1 = self._make_stage(base_channels, base_channels, 2, kernel_size)
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = self._make_stage(base_channels, base_channels * 2, 2, kernel_size)
        self.down2 = nn.MaxPool2d(2)
        self.enc3 = self._make_stage(base_channels * 2, base_channels * 4, 2, kernel_size)
        self.down3 = nn.MaxPool2d(2)
        self.enc4 = self._make_stage(base_channels * 4, base_channels * 8, 2, kernel_size)

        # Decoder stages
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self._make_stage(base_channels * 8, base_channels * 4, 2, kernel_size)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._make_stage(base_channels * 4, base_channels * 2, 2, kernel_size)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._make_stage(base_channels * 2, base_channels, 2, kernel_size)

        # Output head
        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def _make_stage(self, in_ch, out_ch, num_blocks, kernel_size):
        layers = []
        for i in range(num_blocks):
            layers.append(MedNeXtBlock2D(in_ch if i == 0 else out_ch, out_ch, kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        e1 = self.stem(x)
        e1 = self.enc1(e1)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


class MedNeXtBlock2D(nn.Module):
    """2D version of MedNeXt block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels
        )
        self.norm = nn.InstanceNorm2d(in_channels)
        self.act = nn.GELU()
        self.pwconv1 = nn.Conv2d(in_channels, out_channels * 4, 1)
        self.pwconv2 = nn.Conv2d(out_channels * 4, out_channels, 1)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + residual if x.shape == residual.shape else x


class MedNeXtL(nn.Module):
    """
    MedNeXt-L model for cardiac MRI segmentation.

    Configuration: kernel_size=5, 224x224 input, 4 output classes.
    Pre-trained on 319,175 annotated images from 1,555 subjects.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        input_size: int = 224,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.model = MedNeXtEncoder(
            in_channels=in_channels,
            base_channels=32,
            num_classes=num_classes,
            kernel_size=kernel_size,
            input_size=input_size,
        )
        self.input_size = input_size
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 1, H, W) or (B, C, H, W)

        Returns:
            Logits tensor (B, num_classes, H, W)
        """
        return self.model(x)

    def load_checkpoint(self, checkpoint_path: str):
        """Load pre-trained checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'])
        else:
            self.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
