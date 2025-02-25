import pytest
from mnist.dataloader import get_dataloaders

@pytest.fixture
def config():
    return {
        "batch_size": 64,
        "num_workers": 4
    }

def test_dataloaders_creation(config):
    train_loader, test_loader = get_dataloaders(config)
    
    # Check if dataloaders are not None
    assert train_loader is not None
    assert test_loader is not None
    
    # Check if dataloaders have the correct batch size
    assert train_loader.batch_size == config["batch_size"]
    assert test_loader.batch_size == config["batch_size"]
    
    # Check if dataloaders have the correct number of workers
    assert train_loader.num_workers == config["num_workers"]
    assert test_loader.num_workers == config["num_workers"]

def test_dataloaders_data(config):
    train_loader, test_loader = get_dataloaders(config)
    
    # Check if train_loader has data
    train_data_iter = iter(train_loader)
    train_images, train_labels = next(train_data_iter)
    assert train_images.shape[0] == config["batch_size"]
    
    # Check if test_loader has data
    test_data_iter = iter(test_loader)
    test_images, test_labels = next(test_data_iter)
    assert test_images.shape[0] == config["batch_size"]

def test_dataloaders_tensor_shape(config):
    train_loader, test_loader = get_dataloaders(config)
    
    # Check if train_loader yields tensors of the expected shape
    train_data_iter = iter(train_loader)
    train_images, train_labels = next(train_data_iter)
    assert train_images.shape == (config["batch_size"], 1, 28, 28)
    assert train_labels.shape == (config["batch_size"],)
    
    # Check if test_loader yields tensors of the expected shape
    test_data_iter = iter(test_loader)
    test_images, test_labels = next(test_data_iter)
    assert test_images.shape == (config["batch_size"], 1, 28, 28)
    assert test_labels.shape == (config["batch_size"],)