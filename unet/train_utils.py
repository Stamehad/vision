import os
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Print config to screen
    print("\n🔹 Loaded Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    #print("\n")

    return config

def setup_trainer(config, trial_mode=False, profiler=None):
    """Creates a Lightning Trainer with trial mode support and prints checkpoint path."""

    # Logger
    experiment_name = f"UNET_{config['optimizer']['T_max']}epochs"
    logger = TensorBoardLogger("lightning_logs", name=experiment_name)

    # Checkpoint directory
    checkpoint_dir = os.path.join("checkpoints", experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Custom checkpoint filename
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", 
        save_top_k=3,  # Keep top 3 models
        mode="min", 
        dirpath=checkpoint_dir, 
        filename="UNET_epoch={epoch:02d}_val_loss={val_loss:.4f}"  # Custom filename format
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    # Print where checkpoints will be saved
    print(f"\n📂 Checkpoints will be saved in: {os.path.abspath(checkpoint_dir)}\n")
    print(f"📝 Checkpoint file format: {checkpoint_callback.filename}\n")

    trainer_args = {
        "max_epochs": 1 if trial_mode else config["optimizer"]["T_max"],
        "accelerator": "mps" if torch.backends.mps.is_available() else "cpu",
        "logger": logger,
        "callbacks": [checkpoint_callback, early_stop_callback],
        "profiler": profiler,  # Add profiler if available
    }

    if trial_mode:
        trainer_args["limit_train_batches"] = 0.1  # Only use 10% batches
        trainer_args["limit_val_batches"] = 0.1

    return pl.Trainer(**trainer_args), checkpoint_dir