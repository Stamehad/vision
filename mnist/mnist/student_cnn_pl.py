import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from cnn import CNN  # Import the small CNN
from distillation import distillation_loss  # Import our distillation loss
import torchmetrics

class StudentCNN(pl.LightningModule):
    def __init__(self, config, teacher_model):
        super().__init__()
        # unpack config
        self.lr = config["optimizer"]["learning_rate"]
        self.weight_decay = config["optimizer"]["weight_decay"]
        self.max_epochs = config["trainer"].get("max_epochs", 10)
        self.C = config.get("num_classes", 10)
        self.temperature = config["distillation"]["temperature"]
        self.alpha = config["distillation"]["alpha"]

        self.config = config
        self.teacher = teacher_model  # Load pre-trained teacher model
        self.student = CNN(config["model"])  # Small 4K CNN

        # Accuracy metric
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.C)
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.student(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # Forward pass through teacher (without gradients)
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        # Forward pass through student
        student_logits = self(x)

        # Compute knowledge distillation loss
        loss = distillation_loss(student_logits, teacher_logits, y, self.temperature, self.alpha)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

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
    
    # def configure_optimizers(self):
    #     return optim.Adam(self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay)