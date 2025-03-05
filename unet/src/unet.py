import torch
import torch.nn as nn
from unet_parts import EncoderBlock, DecoderBlock

class AttentionUNet(nn.Module):
    """Attention U-Net with encoder-decoder architecture and attention gates."""

    def __init__(self, config):
        super().__init__()

        # unpack the config dictionary
        in_channels = config.get("in_channels", 3)
        out_channels = config.get("out_channels", 3)
        base_filters = config.get("base_filters", 64)

        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_filters)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4)
        self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 16, base_filters * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.dec4 = DecoderBlock(base_filters * 16, base_filters * 8)
        self.dec3 = DecoderBlock(base_filters * 8, base_filters * 4)
        self.dec2 = DecoderBlock(base_filters * 4, base_filters * 2)
        self.dec1 = DecoderBlock(base_filters * 2, base_filters)

        # Final segmentation output
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1_pooled, x1 = self.enc1(x)
        x2_pooled, x2 = self.enc2(x1_pooled)
        x3_pooled, x3 = self.enc3(x2_pooled)
        x4_pooled, x4 = self.enc4(x3_pooled)

        # Bottleneck
        x_b = self.bottleneck(x4_pooled)

        # Decoder (Skip connections with attention)
        x = self.dec4(x_b, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)

        return self.final_conv(x)