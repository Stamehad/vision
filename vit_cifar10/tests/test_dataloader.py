import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

def test_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Ensure normalization is applied
    ])
    
    dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    batch = next(iter(dataloader))
    x, y = batch
    
    assert x.shape == (8, 3, 32, 32), f"Expected (8,3,32,32) but got {x.shape}"
    assert y.shape == (8,), f"Expected (8,) but got {y.shape}"