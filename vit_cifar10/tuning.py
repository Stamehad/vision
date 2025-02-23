import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.tuner.tuning import Tuner
from vit_pl import VIT
from dataloader import get_dataloaders
from train_utils import load_config

def find_learning_rate(config_path):
    """Runs a learning rate finder and suggests an optimal LR."""
    config = load_config(config_path)
    train_loader, test_loader = get_dataloaders(config["training"])
    model = VIT(config)

    trainer = pl.Trainer(max_epochs=1, accelerator="mps" if torch.backends.mps.is_available() else "cpu")
    tuner = Tuner(trainer)

    lr_finder = tuner.lr_find(model, train_dataloaders=train_loader)
    suggested_lr = lr_finder.suggestion()
    
    # Show plot
    fig = lr_finder.plot(suggest=True)
    plt.show()

    print(f"Suggested Learning Rate: {suggested_lr}")
    return suggested_lr

if __name__ == "__main__":
    find_learning_rate("cifar_config.yaml")