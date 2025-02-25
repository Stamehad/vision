import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()

        # Set defaults
        self.n0 = config.get("n0", 1)
        self.n1 = config.get("n1", 32)
        self.n2 = config.get("n2", 64)
        self.k = config.get("kernel_size", 3)
        self.s = config.get("stride", 1)
        self.p = config.get("padding", 1)
        self.C = config.get("num_classes", 10)
        self.H = config.get("n_hidden", 128)
        self.activation = config.get("activation", "relu")

        s_out = config.get("image_size", 28) // 4 

        # Define CNN layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.n0, out_channels=self.n1, kernel_size=self.k, stride=self.s, padding=self.p),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=self.n1, out_channels=self.n2, kernel_size=self.k, stride=self.s, padding=self.p),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1),
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.n2 * s_out * s_out, self.H),
            nn.ReLU(),
            nn.Dropout(0.5),  # ✅ Dropout for regularization
            nn.Linear(self.H, self.C),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

    def summary(self):
        print(self)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_params:,}")