import torch
import yaml
from vit_pl import VIT

def test_training_step():
    # config = {
    #     "model": {"num_classes": 10, "emb_dim": 24},
    #     "training": {"learning_rate": 1e-3, "weight_decay": 1e-4}
    # }

    with open("cifar_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    model = VIT(config)
    
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    batch = (x, y)

    loss = model.training_step(batch, batch_idx=0)
    assert loss > 0, "Loss should be positive"