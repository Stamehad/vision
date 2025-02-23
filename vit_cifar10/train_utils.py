import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_trainer(config, trial_mode=False, profiler=None):
    """Creates a Lightning Trainer with trial mode support."""
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger

    logger = TensorBoardLogger("lightning_logs", name="ViT_CIFAR10")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", save_top_k=1, mode="min", dirpath="checkpoints/", filename="best_model"
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    trainer_args = {
        "max_epochs": 1 if trial_mode else config["training"]["epochs"],
        "accelerator": "mps" if torch.backends.mps.is_available() else "cpu",
        "logger": logger,
        "callbacks": [checkpoint_callback, early_stop_callback],
        "profiler": profiler,  # Add profiler if available
    }

    if trial_mode:
        trainer_args["limit_train_batches"] = 0.1  # Only use 10% batches
        trainer_args["limit_val_batches"] = 0.1

    return pl.Trainer(**trainer_args)