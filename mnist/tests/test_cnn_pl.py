import pytest
import torch
import pytorch_lightning as pl
from mnist.cnn_pl import MNIST_CNN

@pytest.fixture
def config():
    """Fixture to provide a default configuration for the CNN model."""
    return {
        "model": {
            "num_classes": 10,
            "hidden_units": 128,
        },
        "optimizer": {
            "learning_rate": 0.001,
            "weight_decay": 0.0001
        },
        "trainer": {
            "max_epochs": 5
        }
    }

@pytest.fixture
def model(config):
    """Fixture to create the PyTorch Lightning CNN model."""
    return MNIST_CNN(config)

def test_model_initialization(model):
    """Tests that the PyTorch Lightning model initializes without errors."""
    assert model is not None, "Model failed to initialize."

def test_forward_pass(model):
    """Tests that the model forward pass produces correct output shape."""
    batch_size = 8
    input_tensor = torch.randn(batch_size, 1, 28, 28)  # MNIST shape (B, C, H, W)

    output = model(input_tensor)

    # ✅ Expected shape should be (batch_size, num_classes)
    assert output.shape == (batch_size, model.C), f"Expected {(batch_size, model.C)}, but got {output.shape}"

def test_training_step(model):
    """Ensures that the training step runs without crashing."""
    batch_size = 8
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, model.C, (batch_size,))  # Labels (batch_size,)
    
    batch = (x, y)
    try:
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor), "Training step should return a tensor"
    except Exception as e:
        pytest.fail(f"Training step failed with error: {e}")

def test_validation_step(model):
    """Ensures that the validation step runs without crashing and logs accuracy."""
    batch_size = 8
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, model.C, (batch_size,))

    batch = (x, y)
    try:
        loss = model.validation_step(batch, 0)
        assert isinstance(loss, torch.Tensor), "Validation step should return a tensor"
    except Exception as e:
        pytest.fail(f"Validation step failed with error: {e}")

def test_test_step(model):
    """Ensures that the test step runs correctly."""
    batch_size = 8
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, model.C, (batch_size,))

    batch = (x, y)
    try:
        loss = model.test_step(batch, 0)
        assert isinstance(loss, torch.Tensor), "Test step should return a tensor"
    except Exception as e:
        pytest.fail(f"Test step failed with error: {e}")

def test_configure_optimizers(model):
    """Ensures that the optimizer and scheduler are properly configured."""
    try:
        optimizers = model.configure_optimizers()
        assert "optimizer" in optimizers, "Optimizer is missing in configure_optimizers"
        assert "lr_scheduler" in optimizers, "Learning rate scheduler is missing"
    except Exception as e:
        pytest.fail(f"Optimizer configuration failed: {e}")