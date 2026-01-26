#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
python train.py --experiment "cifar100_$(date +%Y%m%d_%H%M%S)" --config ./cifar100_best_cons.yaml --data-path /data/rboone/datasets/cifar100/ --log-wandb --model max_former
