import torch
import torch.nn as nn
import pytorch_lightning as pl
from vit import ViT
import torchmetrics

class VIT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config_train = config["training"]
        num_classes = config["model"].get("num_classes", 10)
        self.lr = self.config_train["learning_rate"]
        
        # Initialize ViT model and loss function
        self.vit = ViT(config["model"])
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Accuracy metric
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.vit(x)

    def training_step(self, batch, batch_idx):
        x, y = self.to_device(*batch)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Log training loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = self.to_device(*batch)
        y_hat = self(x)
        
        loss = self.loss_fn(y_hat, y)
        acc = self.test_acc(y_hat.softmax(dim=1), y)  # Ensure logits → probabilities

        # Log validation loss & accuracy
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = self.to_device(*batch)
        y_hat = self(x) 
        
        loss = self.loss_fn(y_hat, y)
        acc = self.test_acc(y_hat.softmax(dim=1), y)  # Compute accuracy
        
        # Log test loss & accuracy
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_acc", acc, prog_bar=True, logger=True)

        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config_train["learning_rate"],
            weight_decay=self.config_train["weight_decay"]
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config_train.get("epochs", 10)
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def to_device(self, x, y):
        device = self.device
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)