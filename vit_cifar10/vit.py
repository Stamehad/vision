import torch.nn as nn
from modules import PatchEmbedding, TransformerEncoder, ClassificationHead

class ViT(nn.Module):
    def __init__(self, config, verbose=False): 
        super(ViT, self).__init__()
        """
        Vision Transformer (ViT) Model.
        Architecture:
            1. Patch Embedding: linear projection of image patches + positional embeddings
            2. Transformer Encoder
            3. Classification Head: layer norm, dropout, linear projection
        Args:
            config disctionary:
            img_size: Input image size. Default: 32
            patch_size: Patch size. Default: 4
            in_channels: Number of input channels. Default: 3
            emb_dim: Embedding dimension. Default: 128
            num_heads: Number of attention heads. Default: 8
            num_layers: Number of transformer layers. Default: 6
            mlp_ratio: Multiplier for MLP dimension. Default: 4
            num_classes: Number of classes in the dataset.
            dropout: Dropout probability. Default: 0.1
        Forward pass:
            x: Input tensor of shape (B, C, H, W)
            logits: Output logits of shape (B, num_classes)
        """

        # Store config for easy debugging
        self.config = config  
        num_layers = config.get("num_layers", 6)
        num_heads = config.get("num_heads", 8)
        emb_dim = config.get("emb_dim", 128)

        # Initialize components
        self.patch_embed = PatchEmbedding(config)
        self.transformer = TransformerEncoder(config)
        self.class_head = ClassificationHead(config)

        # Print Model Summary
        if verbose:
            print(f"ViT Model: {num_layers} layers, {num_heads} heads, emb_dim={emb_dim}")

    def forward(self, x):
        x = self.patch_embed(x) 
        x = self.transformer(x)
        x = self.class_head(x)
        return x
        