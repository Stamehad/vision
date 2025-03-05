import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """Attention Gate to refine skip connection features before merging into the decoder."""
    
    def __init__(self, F_g, F_l, F_int):
        """
        F_g  = Number of filters in the decoder feature map (gate signal).
        F_l  = Number of filters in the encoder skip connection.
        F_int = Intermediate feature map size (usually smaller).
        """
        super(AttentionGate, self).__init__()

        # 1x1 Conv layers to match feature dimensions
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)

        # Attention map (sigmoid activation)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        """
        x = Encoder skip connection feature map.
        g = Decoder feature map (gate signal).
        """
        # Linear transformations
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Sum + ReLU activation
        psi = F.relu(g1 + x1)

        # Generate attention map
        psi = self.sigmoid(self.psi(psi))

        # Apply attention map to the skip connection
        return x * psi