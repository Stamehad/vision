import torch
import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from train_utils import load_config, setup_trainer
from src.dataloader import get_dataloaders
from src.model_pl import UNetPL

def main():
    # Load model config
    config = load_config("unet_config.yaml")

    # Modify data config for quick testing
    config["data"]["batch_size"] = 4  # Use a small batch size
    config["trainer"]["limit_train_batches"] = 0.1  # Use 10% of training data
    config["trainer"]["limit_val_batches"] = 0.1  # Use 10% of validation data
    config["trainer"]["max_epochs"] = 2  # Only 2 epochs for quick test

    # Get DataLoaders
    train_loader, val_loader = get_dataloaders(config["data"])

    # Initialize Model
    model = UNetPL(config)

    # Use PyTorch Profiler to analyze performance
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("lightning_logs/profiler"),
        schedule=torch.profiler.schedule(
            wait=2,   # Skip first 2 steps
            warmup=3,  # Warm up next 3 steps
            active=5,  # Profile for 5 steps
            repeat=1
        ),
        with_stack=True
    )

    # Set up Trainer for a quick trial
    trainer, checkpoint_dir = setup_trainer(config, trial_mode=True, profiler=profiler)

    print("🚀 Running a trial with 2 epochs and 10% batches...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(profiler.summary())

if __name__ == "__main__":
    main()