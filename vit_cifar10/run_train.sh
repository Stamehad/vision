#!/bin/bash

# Activate conda environment
source ~/miniconda3/bin/activate my_vision

# Run training
python train.py --config cifar_config.yaml --epochs 8 --checkpoint /vit_cifar10/lightning_logs/ViT_CIFAR10/version_10/checkpoints/epoch=3-step=1564.ckpt # --batch_size 64 --lr 0.001 --num_workers 8

# Start TensorBoard
tensorboard --logdir "lightning_logs" --port 6006