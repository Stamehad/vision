import torch
import torch.nn as nn
from unet_parts import EncoderBlock, DecoderBlock
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../vit_cifar10')))
from utils import debug_hook

class TinyUNet(nn.Module):
    """A lightweight U-Net with attention gates optimized for MPS/CPU."""

    def __init__(self, config) -> None:
        super().__init__()

        # unpack the config dictionary
        in_channels = config.get("in_channels", 3)
        out_channels = config.get("out_channels", 3)
        base_filters = config.get("base_filters", 32)

        # Encoder (Downsampling)
        self.enc1 = EncoderBlock(in_channels, base_filters)  # (B, 3, 128, 128) → (B, 32, 64, 64)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2)  # (B, 32, 64, 64) → (B, 64, 32, 32)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4)  # (B, 64, 32, 32) → (B, 128, 16, 16)

        # Bottleneck (No further downsampling)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )  # (B, 128, 16, 16) → (B, 128, 16, 16)

        # Decoder (Upsampling)
        # self.dec3 = DecoderBlock(base_filters * 4, base_filters * 4)  # (B, 128, 16, 16) → (B, 64, 32, 32)
        # self.dec2 = DecoderBlock(base_filters * 4, base_filters * 2)  # (B, 64, 32, 32) → (B, 32, 64, 64)

        self.dec3 = DecoderBlock(base_filters * 4, base_filters * 4)  # (B, 128, 16, 16) → (B, 64, 32, 32)
        self.dec2 = DecoderBlock(base_filters * 4, base_filters * 2)  # (B, 64, 32, 32) → (B, 32, 64, 64)
        self.dec1 = DecoderBlock(base_filters * 2, base_filters)      # (B, 32, 64, 64) → (B, 16, 128, 128)

        # Final segmentation output
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)  # (B, 32, 64, 64) → (B, 3, 64, 64)

    @debug_hook
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Tiny U-Net."""
        # Encoder path
        x1_pooled, x1 = self.enc1(x)  # (B, 3, 128, 128) → (B, 32, 64, 64)
        x2_pooled, x2 = self.enc2(x1_pooled)  # (B, 32, 64, 64) → (B, 64, 32, 32)
        x3_pooled, x3 = self.enc3(x2_pooled)  # (B, 64, 32, 32) → (B, 128, 16, 16)

        # Bottleneck
        x_b = self.bottleneck(x3_pooled)

        # Decoder path (Skip connections)
        x = self.dec3(x_b, x3)  # (B, 128, 16, 16) → (B, 64, 32, 32)
        x = self.dec2(x, x2)  # (B, 64, 32, 32) → (B, 32, 64, 64)
        x = self.dec1(x, x1)  # (B, 32, 64, 64) → (B, 16, 128, 128)

        return self.final_conv(x)  # (B, 32, 64, 64) → (B, 3, 64, 64)