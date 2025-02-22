import torch
import yaml
# import pytest
from vit_cifar10.vit import ViT

def test_vit_forward_pass():
    # config = {
    #     "img_size": 32,
    #     "patch_size": 2,
    #     "in_channels": 3,
    #     "emb_dim": 24,
    #     "num_heads": 3,
    #     "num_classes": 10,
    #     "mlp_ratio": 4,
    #     "num_layers": 6,
    #     "dropout": 0.1,
    # }

    with open("cifar_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    
    model = ViT(config["model"])
    x = torch.randn(4, 3, 32, 32)  # Batch size = 4
    y = model(x)
    
    assert y.shape == (4, 10), f"Expected shape (4,10), but got {y.shape}"