#!/bin/bash
# Configuration 2 - Best settings from event_stable_sweep
# Key characteristics:
#   - Lower alpha/beta values (1.336/1.298)
#   - Higher k-dir (2.911)
#   - Lower snnbp-max-ratio (2.032)
#   - Lower weight decay (0.0025)

export CUDA_VISIBLE_DEVICES=1
python train.py \
    --log-wandb \
    --config=./event_sweepcfile.yaml \
    --dS-du=Gamma \
    --data-path=/data/rboone/datasets/wg_dvst \
    --dataset=cifar10dvs \
    --du-du=conservative_cgrad \
    --epochs=96 \
    --experiment=event_config2_eval \
    --lr=0.003162790343105352 \
    --model=max_former \
    --snnbp-alpha=1.3363409750901318 \
    --snnbp-beta=1.2982628293585403 \
    --snnbp-decay=0.38512368739106906 \
    --snnbp-epsilon=0.13082664029692248 \
    --snnbp-intervention=0.5318500985427234 \
    --snnbp-k-dir=2.910821617960099 \
    --snnbp-max-ratio=2.03242114666679 \
    --weight-decay=0.0025388144945772927 \
    --early-stop-patience=-1
