#!/bin/bash
# Configuration 1 - Best settings from event_stable_sweep
# Key characteristics:
#   - Higher alpha/beta values (2.006/3.639)
#   - Higher k-dir (0.745)
#   - Higher snnbp-max-ratio (3.625)
#   - Higher weight decay (0.0201)

export CUDA_VISIBLE_DEVICES=0
python train.py \
    --log-wandb \
    --config=./event_sweepcfile.yaml \
    --dS-du=Gamma \
    --data-path=/data/rboone/datasets/wg_dvst \
    --dataset=cifar10dvs \
    --du-du=conservative_cgrad \
    --epochs=96 \
    --experiment=event_config1_eval \
    --lr=0.003143331697249924 \
    --model=max_former \
    --snnbp-alpha=2.006323097077301 \
    --snnbp-beta=3.639293046797402 \
    --snnbp-decay=0.4156090921944937 \
    --snnbp-epsilon=0.28756756400199823 \
    --snnbp-intervention=0.6120316105184137 \
    --snnbp-k-dir=0.745393025395608 \
    --snnbp-max-ratio=3.6249955929322 \
    --weight-decay=0.0201245873809068 \
    --early-stop-patience=-1
