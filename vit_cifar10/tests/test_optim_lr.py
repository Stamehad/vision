import yaml
from vit_cifar10.vit_pl import VIT 

def test_optimizer():
    # config = {
    #     "model": {"num_classes": 10, "emb_dim": 24},
    #     "training": {"learning_rate": 1e-3, "weight_decay": 1e-4, "epochs": 10}
    # }

    with open("cifar_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    model = VIT(config)
    optimizers = model.configure_optimizers()

    assert "optimizer" in optimizers, "Optimizer should be defined"
    assert "lr_scheduler" in optimizers, "Learning rate scheduler should be defined"