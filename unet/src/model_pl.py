import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from torchmetrics import Dice, Accuracy

from src.unet_tiny import TinyUNet

def multiclass_dice_score(preds, targets, num_classes=3, epsilon=1e-6):
    """Computes Dice Score for multi-class segmentation."""
    dice_scores = []
    
    for class_idx in range(num_classes):
        pred_mask = (preds == class_idx).float()
        true_mask = (targets == class_idx).float()

        intersection = (pred_mask * true_mask).sum()
        union = pred_mask.sum() + true_mask.sum()

        dice = (2. * intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice)

    return torch.mean(torch.stack(dice_scores))  # Average over all classes

def multiclass_accuracy(preds, targets):
    """Computes accuracy for multi-class segmentation."""
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total

def get_metrics(y_hat, y):
    """Returns Dice Score and Accuracy for multi-class segmentation."""
    # Convert logits to class indices
    y_hat_class = torch.argmax(y_hat, dim=1)  # Shape: (B, H, W)
    y = y.long()  # Ensure targets are long type

    # Compute Dice Score and Accuracy
    dice = multiclass_dice_score(y_hat_class, y)
    acc = multiclass_accuracy(y_hat_class, y)
    return dice, acc

class UNetPL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.lr = config["optimizer"]["learning_rate"]
        self.weight_decay = config["optimizer"]["weight_decay"]
        self.T_max = config["optimizer"].get("T_max", 50)

        self.save_hyperparameters(config)  # Save config automatically
        self.model = TinyUNet(config["model"])

        self.criterion = nn.CrossEntropyLoss()  # Multi-class Segmentation Loss

    def forward(self, x):
        return self.model(x.to(self.device))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1).long().to(self.device)  # Ensure shape is (B, H, W)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Compute metrics
        dice_score, acc = get_metrics(y_hat, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({"train_dice": dice_score, "train_acc": acc}, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1).long().to(self.device)  # Ensure shape is (B, H, W)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Compute metrics
        dice_score, acc = get_metrics(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({"val_dice": dice_score, "val_acc": acc}, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1).long().to(self.device)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Compute metrics
        dice_score, acc = get_metrics(y_hat, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({"test_dice": dice_score, "test_acc": acc}, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        