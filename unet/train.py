import os
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import argparse

from src.dataloader import get_dataloaders
from src.model_pl import UNetPL
from train_utils import load_config, setup_trainer

# setup arg parser to get checkpoint path
def parse_args():
    parser = argparse.ArgumentParser(description="UNet for Image Segmentation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint file to resume training")
    return parser.parse_args()

def main():
    # Load args
    args = parse_args()
    #config = load_config(args.config)

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
    trainer, checkpoint_dir = setup_trainer(config) #, profiler="simple")

    # Initial test
    #trainer.test(model, val_loader)

    # Start Training
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.checkpoint)

    trainer.test(model, val_loader)  # Run test set

    # Print final checkpoint path
    print(f"\n✅ Training Complete! Best checkpoint saved in: {checkpoint_dir}\n")

if __name__ == "__main__":
    main()