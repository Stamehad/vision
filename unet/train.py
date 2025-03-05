import os
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataloader import get_dataloaders
from src.model_pl import UNetPL
from train_utils import load_config, setup_trainer

def main():
    # Load YAML config
    config_path = "unet_config.yaml"
    config = load_config(config_path)

    # Set random seed for reproducibility
    pl.seed_everything(config["training"].get("seed", 42))

    # Get DataLoaders
    train_loader, val_loader = get_dataloaders(config["data"])

    # Initialize Model
    model = UNetPL(config)

    # Setup Trainer
    trainer, checkpoint_dir = setup_trainer(config)

    # Start Training
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.test(model, val_loader)  # Run test set

    # Print final checkpoint path
    print(f"\n✅ Training Complete! Best checkpoint saved in: {checkpoint_dir}\n")

if __name__ == "__main__":
    main()