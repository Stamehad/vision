import torch
import torch.nn as nn
from src.attention_gate import AttentionGate

class EncoderBlock(nn.Module):
    """Encoder block with two conv layers + max pooling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

    def forward(self, x):                   # (B, C, H, W) = (B, C_in, H, W)
        x_out = self.conv(x)                # → (B, C_out, H, W)
        x_pooled = self.pool(x_out)         # → (B, C_out, H/2, W/2)
        return x_pooled, x_out  # Pooled output + skip connection output

class DecoderBlock(nn.Module):
    """Decoder block with upsampling + attention + convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):  # x = (B, C_in, H, W), skip = (B, C_skip, H/2, W/2)
        x = self.upconv(x)       # → (B, C_out, H, W)
        skip = self.attention(skip, x)  # Apply attention before concatenation
        x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
        return self.conv(x)