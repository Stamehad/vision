import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from vit_pl import VIT
from dataloader import get_dataloaders
from train_utils import load_config, setup_trainer

# 1. Setup Argparse for CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vision Transformer (ViT) on CIFAR-10")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint file to resume training")
    return parser.parse_args()

def main():
    args = parse_args()  # Read CLI args
    config = load_config(args.config)  # Load config

    # Override config with command-line args
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    # Load data
    train_loader, test_loader = get_dataloaders(config["training"])

    # Initialize model
    model = VIT(config)

    # Initialize trainer
    trainer = setup_trainer(config)

    # Train (Resume if checkpoint is given)
    trainer.fit(model, train_loader, test_loader, ckpt_path=args.checkpoint)

    # Evaluate after training
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()