import argparse
import torch
import glob
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from cnn import CNN
from student_cnn_pl import StudentCNN
from cnn_pl import MNIST_CNN  # Import LightningModule instead
from dataloader import get_dataloaders
from train_utils import load_config, setup_trainer

# # Default teacher checkpoint path
# TEACHER_CHECKPOINT_PATH = "/Users/itamarshamir/Documents/code/Vision/mnist/mnist/checkpoints/MNIST_CNN_8epochs/MNIST_CNN_epoch=epoch=07_val_loss=val_loss=0.0251.ckpt"

# 1. Setup Argparse for CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train MNIST Student CNN with Knowledge Distillation")
    parser.add_argument("--config", type=str, default="configs/mnist_config.yaml", help="Path to config file")
    parser.add_argument("--teacher_config", type=str, default="configs/mnist_config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a student checkpoint to resume training")
    parser.add_argument("--teacher_checkpoint", type=str, default=None, help="Path to a student checkpoint to resume training")
    parser.add_argument("--trial", type=bool, default=False, help="Run in trial mode")
    return parser.parse_args()

def main():
    args = parse_args()  # Read CLI args
    config = load_config(args.config)  # Load config
    teacher_config = load_config(args.teacher_config)  # Load teacher config

    # Override config with command-line args
    if args.epochs:
        config["trainer"]["max_epochs"] = args.epochs
    if args.batch_size:
        config["dataset"]["batch_size"] = args.batch_size
    if args.lr:
        config["optimizer"]["learning_rate"] = args.lr

    # Load data
    train_loader, test_loader = get_dataloaders(config["dataset"])

    # Load teacher model
    teacher_model = MNIST_CNN(teacher_config)
    teacher_checkpoint = args.teacher_checkpoint

    teacher_checkpoint_dict = torch.load(teacher_checkpoint, map_location=torch.device("cpu"), weights_only=True)
    teacher_model.load_state_dict(teacher_checkpoint_dict["state_dict"])

    # teacher_model.load_state_dict(torch.load(teacher_checkpoint, map_location=torch.device("cpu")))
    teacher_model.eval()  # Make sure it's in eval mode

    print(f"\n🧑‍🏫 Teacher Model Loaded from: {teacher_checkpoint}\n")

    # Initialize student model
    student_model = StudentCNN(config, teacher_model)

    # Initialize trainer
    trainer, checkpoint_dir = setup_trainer(config, trial_mode=args.trial)

    # Train the student model (Resume if checkpoint is given)
    trainer.fit(student_model, train_loader, test_loader, ckpt_path=args.checkpoint)

    # Evaluate after training
    trainer.test(student_model, test_loader)

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