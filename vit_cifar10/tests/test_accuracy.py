import torch 
import yaml
from vit_cifar10.vit_pl import VIT

def test_accuracy():
    config = {
        "model": {"num_classes": 10, "emb_dim": 24},
        "training": {"learning_rate": 1e-3, "weight_decay": 1e-4}
    }

    with open("cifar_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    model = VIT(config)

    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    batch = (x, y)

    test_out = model.test_step(batch, batch_idx=0)
    
    assert "test_acc" in test_out, "Test step should return accuracy"
    assert 0 <= test_out["test_acc"] <= 1, "Accuracy should be between 0 and 1"