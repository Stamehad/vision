import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl

from src.unet_tiny import TinyUNet

def binary_dice_score(preds, targets, epsilon=1e-6):
    """Computes Dice Score for binary segmentation."""
    preds = (preds > 0.5).float()  # Convert to binary predictions
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice

def binary_accuracy(preds, targets):
    """Computes accuracy for binary segmentation."""
    preds = (preds > 0.5).float()  # Convert logits to binary (0 or 1)
    return (preds == targets).float().mean().item()
    # correct = (preds == targets).sum().item()
    # total = targets.numel()
    # return correct / total

class UNetPL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.lr = config["optimizer"]["learning_rate"]
        self.weight_decay = config["optimizer"]["weight_decay"]
        self.T_max = config["optimizer"].get("T_max", 50)

        self.save_hyperparameters(config)
        self.model = TinyUNet(config["model"])  # Still using TinyUNet

        self.criterion = nn.BCEWithLogitsLoss()  # ✅ Use binary loss

    def forward(self, x):
        return self.model(x.to(self.device))  # Shape: (B, 1, H, W)

    def training_step(self, batch, batch_idx):
        x, y = batch # x.shape = (B, 3, H, W), y.shape = (B, 1, H, W)
        y = y.squeeze(1).float().to(self.device)  # Convert to float (NOT long)
        y = (y > 0).float()  # Converts 0 → 0 and anything else → 1

        y_hat = self(x).squeeze(1)  # Remove channel dim
        loss = self.criterion(y_hat, y)  # Binary loss

        # Apply sigmoid for probability outputs
        y_hat_prob = torch.sigmoid(y_hat)

        # Compute metrics
        dice_score = binary_dice_score(y_hat_prob, y)
        acc = binary_accuracy(y_hat_prob, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({"train_dice": dice_score, "train_acc": acc}, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1).float().to(self.device)  # Convert to float (NOT long)
        y = (y > 0).float()  # Converts 0 → 0 and anything else → 1

        y_hat = self(x).squeeze(1)
        loss = self.criterion(y_hat, y)

        y_hat_prob = torch.sigmoid(y_hat)
        dice_score = binary_dice_score(y_hat_prob, y)
        acc = binary_accuracy(y_hat_prob, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({"val_dice": dice_score, "val_acc": acc}, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1).float().to(self.device)  # Convert to float (NOT long)
        y = (y > 0).float()  # Converts 0 → 0 and anything else → 1

        y_hat = self(x).squeeze(1)
        loss = self.criterion(y_hat, y)

        y_hat_prob = torch.sigmoid(y_hat)
        dice_score = binary_dice_score(y_hat_prob, y)
        acc = binary_accuracy(y_hat_prob, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({"test_dice": dice_score, "test_acc": acc}, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}