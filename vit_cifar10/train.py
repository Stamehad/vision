import torch
import pytorch_lightning as pl
import yaml
import argparse
from vit_pl import VIT
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision import datasets, transforms
from dataloader import get_dataloaders
from pytorch_lightning.loggers import TensorBoardLogger

# 1. Function to Load Config File
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# 2. Setup Argparse for CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vision Transformer (ViT) on CIFAR-10")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    return parser.parse_args()

# 3. Main Training Function
def main():
    args = parse_args()  # Read CLI args
    config = load_config(args.config)  # Load config from YAML

    # Override config with command-line args
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    # Load Dataloaders from `dataloader.py`
    train_loader, test_loader = get_dataloaders(config["training"])

    torch.backends.mps.is_available()
    torch.backends.mps.is_built()

    torch.backends.mps.allow_tf32 = True  # Allow TensorFloat32 for better performance
    #torch.backends.mps.optimize()  # Optimize MPS backend

    # Initialize model
    model = VIT(config)
    # model = torch.compile(model) does not work with mps

    # Callbacks for Checkpointing & Early Stopping
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    # Add this to trainer in train.py
    logger = TensorBoardLogger("lightning_logs", name="ViT_CIFAR10")

    # Set up Trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        logger=logger,  # Enable TensorBoard logging
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    # Train
    trainer.fit(model, train_loader, test_loader)

    # Test after training
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()