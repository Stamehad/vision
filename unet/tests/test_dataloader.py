import pytest
import torch
from src.dataloader import get_dataloaders

@pytest.fixture
def dataloaders():
    config = {
        "batch_size": 4,
        "num_workers": 2
    }
    return get_dataloaders(config)

def test_dataloaders_creation(dataloaders):
    """Test if dataloaders are successfully created."""
    train_loader, test_loader = dataloaders

    assert train_loader is not None
    assert test_loader is not None

def test_dataloader_shapes(dataloaders):
    """Test if images and masks have correct shapes."""
    train_loader, test_loader = dataloaders
    images, masks = next(iter(train_loader))

    assert images.shape == (4, 3, 128, 128)  # Batch_size=4, 3 RGB channels, 128x128
    assert masks.shape == (4, 1, 128, 128)  # Batch_size=4, 1 mask channel, 128x128

def test_dataloader_non_empty(dataloaders):
    """Ensure the dataset contains samples."""
    train_loader, test_loader = dataloaders

    assert len(train_loader.dataset) > 0
    assert len(test_loader.dataset) > 0