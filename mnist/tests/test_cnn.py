import pytest
import torch
from mnist.cnn import CNN

@pytest.fixture
def config():
    """Fixture to provide a default configuration for the CNN model."""
    return {
        "n0": 1,  # Input channels (for grayscale images)
        "n1": 32,  # First conv layer channels
        "n2": 64,  # Second conv layer channels
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "num_classes": 10,  # Number of output classes
        "n_hidden": 128,  # Hidden layer size
        "image_size": 28  # Expected image size (for MNIST)
    }

@pytest.fixture
def model(config):
    """Fixture to create the CNN model."""
    return CNN(config)

def test_cnn_output_shape(model, config):
    """Tests that the CNN produces the correct output shape given an input tensor."""
    batch_size = 16
    input_tensor = torch.randn(batch_size, config["n0"], config["image_size"], config["image_size"])  # (B, C, H, W)
    
    output = model(input_tensor)

    # Expected shape should be (batch_size, num_classes)
    assert output.shape == (batch_size, config["num_classes"]), f"Expected shape {(batch_size, config['num_classes'])}, but got {output.shape}"

def test_cnn_forward_pass(model, config):
    """Ensures the CNN's forward pass runs without errors."""
    batch_size = 8
    input_tensor = torch.randn(batch_size, config["n0"], config["image_size"], config["image_size"])  # (B, C, H, W)
    
    try:
        output = model(input_tensor)
    except Exception as e:
        pytest.fail(f"Forward pass failed with error: {e}")

def test_cnn_parameter_count(model):
    """Tests that the model has a reasonable number of trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params > 10_000, f"Model has too few parameters: {total_params}"