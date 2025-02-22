import torch
import torch.nn as nn
from utils import debug_hook

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        """
        Patch Embedding module for VisionTransformer.
        Args:
            config["img_size"]: Size of input image. Default: 32
            config["patch_size"]: Size of each patch. Default: 2
            config["in_channels"]: Number of input channels. Default: 3
            config["emb_dim"]: Embedding dimension. Default: 24
        """
        # unpack config dictionary
        img_size = config.get("img_size", 32)
        patch_size = config.get("patch_size", 2)
        in_channels = config.get("in_channels", 3)
        emb_dim = config.get("emb_dim", 24)

        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 256 for 2x2 patches on 32x32
        self.linear_projection = nn.Linear(in_channels * patch_size * patch_size, emb_dim)

        # Special classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # Learnable position embeddings
        self.positional_emb = nn.Embedding(self.n_patches + 1, emb_dim)

        # Register position IDs so they move with model
        self.register_buffer("position_ids", torch.arange(self.n_patches + 1).unsqueeze(0))

    @debug_hook
    def forward(self, x):  # x: (B, 3, 32, 32)
        B, C, H, W = x.shape  # (Batch, Channels, Height, Width)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, 16, 16, 3, 2, 2)
        x = x.view(B, self.n_patches, -1)  # (B, 256, 12)
        x = self.linear_projection(x)  # (B, 256, emb_dim)

        # Prepend the CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Expand to (B, 1, emb_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 257, emb_dim)

        # Add position embeddings (expanded across batch)
        pos_emb = self.positional_emb(self.position_ids).expand(B, -1, -1)  # (B, 257, emb_dim)
        x += pos_emb
        return x
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, config): 
        super().__init__()
        """
        Transformer Encoder Block for VisionTransformer.
        Args:
            config["emb_dim"]: Embedding dimension. 
            config["num_heads"]: Number of attention heads.
            config["mlp_ratio"]: Multiplier for MLP dimension. Default: 4
            config["dropout"]: Dropout probability. Default: 0.1
        """
        # unpack config dictionary
        emb_dim = config["emb_dim"]
        num_heads = config["num_heads"]
        mlp_ratio = config.get("mlp_ratio", 4)
        dropout = config.get("dropout", 0.1)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_ratio * emb_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * emb_dim, emb_dim),
            nn.Dropout(dropout)
        )
        

    def forward(self, x):  # x: (B, N, emb_dim)
        # Apply LayerNorm once instead of 3 times
        x_norm = self.norm1(x)

        # Multi-Head Self-Attention
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)  # (B, N, emb_dim)
        x = x + self.dropout1(attn_output) 

        # Feedforward Network (MLP)
        x = x + self.mlp(self.norm2(x)) 

        return x  # (B, N, emb_dim)
    
class TransformerEncoder(nn.Module):
    def __init__(self, config): # emb_dim, num_heads, num_layers, mlp_ratio=4, dropout=0.1):
        super().__init__()
        """
        Transformer Encoder for VisionTransformer.
        Args:
            config["emb_dim"]: Embedding dimension. 
            config["num_heads"]: Number of attention heads.
            config["num_layers"]: Number of transformer layers.
            config["mlp_ratio"]: Multiplier for MLP dimension.
            config["dropout"]: Dropout probability. Default: 0.1
        """
        # unpack config dictionary
        num_layers = config.get("num_layers", 6)
        
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(config)
            for _ in range(num_layers)
        ])
    
    @debug_hook
    def forward(self, x):  # x: (B, N, d_hidden)
        for layer in self.layers:
            x = layer(x)
        return x  # (B, N, d_hidden)

class ClassificationHead(nn.Module):
    def __init__(self, config): # emb_dim, num_classes, dropout=0.1):
        super().__init__()
        """
        Classification head for VisionTransformer.
        Args:
            config["emb_dim"]: Embedding dimension. 
            config["num_classes"]: Number of classes in the dataset.
            config["dropout"]: Dropout probability. Default: 0.1
        """
        # unpack config dictionary
        emb_dim = config["emb_dim"]
        num_classes = config["num_classes"]
        dropout = config.get("dropout", 0.1)
    
        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Dropout(dropout),  # Regularization
            nn.Linear(emb_dim, num_classes)
        )
        
    @debug_hook
    def forward(self, x):  # x: (B, N, emb_dim)
        return self.mlp(x[:, 0, :])  # Extract CLS token (B, num_classes)