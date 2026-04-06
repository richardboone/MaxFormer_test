#!/bin/bash
# Advanced comparison script with multiple seeds for statistical significance
# Usage: ./run_comparison.sh [num_seeds] [gpu_id]
# Example: ./run_comparison.sh 3 cuda:0

NUM_SEEDS=${1:-3}
GPU=${2:-cuda:0}

SEEDS=(42 1337 2024)

echo "=========================================="
echo "Multi-seed Evaluation for Statistical Significance"
echo "Number of seeds: $NUM_SEEDS"
echo "GPU: $GPU"
echo "=========================================="
echo ""

for ((i=0; i<NUM_SEEDS && i<${#SEEDS[@]}; i++)); do
    SEED=${SEEDS[$i]}
    echo "=========================================="
    echo "Seed $((i+1))/$NUM_SEEDS: $SEED"
    echo "=========================================="
    
    # Config 1 with seed
    echo "[Config 1] Running with seed $SEED..."
    python train.py \
        --log-wandb \
        --config=./event_sweepcfile.yaml \
        --dS-du=Gamma \
        --data-path=/data/rboone/datasets/wg_dvst \
        --dataset=cifar10dvs \
        --du-du=conservative_cgrad \
        --epochs=96 \
        --experiment=config1_seed${SEED} \
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
        --seed=$SEED \
        --device=$GPU
    
    # Config 2 with seed
    echo "[Config 2] Running with seed $SEED..."
    python train.py \
        --log-wandb \
        --config=./event_sweepcfile.yaml \
        --dS-du=Gamma \
        --data-path=/data/rboone/datasets/wg_dvst \
        --dataset=cifar10dvs \
        --du-du=conservative_cgrad \
        --epochs=96 \
        --experiment=config2_seed${SEED} \
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
        --seed=$SEED \
        --device=$GPU
done

echo ""
echo "=========================================="
echo "Multi-seed evaluation complete!"
echo "Compare results in wandb by grouping experiment names"
echo "=========================================="
