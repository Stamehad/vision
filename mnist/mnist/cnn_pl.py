import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from cnn import CNN
import torchmetrics

class MNIST_CNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        # unpack config
        self.lr = config["optimizer"]["learning_rate"]
        self.weight_decay = config["optimizer"]["weight_decay"]
        self.max_epochs = config["trainer"].get("max_epochs", 10)
        self.C = config.get("num_classes", 10)

        # Define CNN
        self.cnn = CNN(config["model"])

        # Accuracy metric
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.C)

    def forward(self, x):                   # x: (B, 1, 28, 28)
        return self.cnn(x)                  # -> (B, 10)

    def training_step(self, batch, batch_idx):
        x, y = self._to_device(*batch)
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        self.log_dict({"train_loss": loss}, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._to_device(*batch)
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)        
        acc = self.test_acc(y_hat.softmax(dim=1), y)  # Ensure logits → probabilities

        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = self._to_device(*batch)
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)  # Compute loss
        acc = self.test_acc(y_hat.softmax(dim=1), y)  # Ensure logits → probabilities

        self.log_dict({"test_loss": loss, "test_acc": acc}, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def _to_device(self, x, y):
        device = self.device
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)