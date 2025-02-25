import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from cnn_pl import MNIST_CNN
from dataloader import get_dataloaders
from train_utils import load_config, setup_trainer
import glob

# 1. Setup Argparse for CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train MNIST CNN model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint file to resume training")
    parser.add_argument("--trial", type=bool, default=False, help="Run in trial mode")
    return parser.parse_args()

def main():
    args = parse_args()  # Read CLI args
    config = load_config(args.config)  # Load config

    # Override config with command-line args
    if args.epochs:
        config["trainer"]["max_epochs"] = args.epochs
    if args.batch_size:
        config["dataset"]["batch_size"] = args.batch_size
    if args.lr:
        config["optimizer"]["learning_rate"] = args.lr

    # Load data
    train_loader, test_loader = get_dataloaders(config["dataset"])

    # Initialize model
    model = MNIST_CNN(config)

    # Initialize trainer
    trainer, checkpoint_dir = setup_trainer(config, trial_mode=args.trial)

    trainer.test(model, test_loader, ckpt_path=args.checkpoint)

    # Train (Resume if checkpoint is given)
    trainer.fit(model, train_loader, test_loader, ckpt_path=args.checkpoint)

    # Evaluate after training
    trainer.test(model, test_loader)

    # 📌 Print all saved checkpoints
    saved_checkpoints = glob.glob(f"{checkpoint_dir}/*.ckpt")
    if saved_checkpoints:
        print("\n✅ Training complete. Checkpoints saved:\n")
        for checkpoint in saved_checkpoints:
            print(f"📌 {checkpoint}")
    else:
        print("\n⚠️ No checkpoints were saved!\n")

if __name__ == "__main__":
    main()