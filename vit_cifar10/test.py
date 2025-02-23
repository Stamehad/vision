import torch
import argparse
import pytorch_lightning as pl
from vit_pl import VIT
from dataloader import get_dataloaders
from train_utils import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained ViT model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    _, test_loader = get_dataloaders(config["training"])

    model = VIT.load_from_checkpoint(args.checkpoint, config=config)

    trainer = pl.Trainer(accelerator="mps" if torch.backends.mps.is_available() else "cpu")
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()