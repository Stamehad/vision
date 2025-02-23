import torch
import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from train_utils import load_config, setup_trainer
from dataloader import get_dataloaders
from vit_pl import VIT

def main():
    config = load_config("cifar_config.yaml")  # Load model config
    train_loader, test_loader = get_dataloaders(config["training"])

    model = VIT(config)

    # Use PyTorch Profiler to analyze performance
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("lightning_logs/profiler"),
        schedule=torch.profiler.schedule(
            wait=2,  # Skip first 2 steps
            warmup=3,  # Warmup next 3 steps
            active=5,  # Profile for 5 steps
            repeat=1
        ),
        with_stack=True
    )

    trainer = setup_trainer(config, trial_mode=True, profiler=profiler)  # Trial mode ON

    print("🚀 Running a trial with 1 epoch and 10% batches...")
    trainer.fit(model, train_loader, test_loader)

    print(profiler.summary())

if __name__ == "__main__":
    main()