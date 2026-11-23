#!/bin/bash
# ==============================================================================
# 4 Full MaxFormer Training Runs on CIFAR10-DVS with WandB Logging
#
# Matrix of 4 configurations:
#   1. LIF   + PyTorch Autograd (Triton OFF)
#   2. LIF   + Fused Triton GPU Kernels (Triton ON)
#   3. CGrad + PyTorch Autograd (Triton OFF)
#   4. CGrad + Fused Triton GPU Kernels (Triton ON)
#
# Usage (from inside the event/ directory):
#   ./run_4_benchmarks.sh sequential [GPU_ID]         # e.g. ./run_4_benchmarks.sh sequential 2
#   ./run_4_benchmarks.sh parallel [GPU_START]        # e.g. ./run_4_benchmarks.sh parallel 2 (uses GPUs 2,3,4,5)
#   ./run_4_benchmarks.sh run_triton_only [GPU1] [GPU2] # e.g. ./run_4_benchmarks.sh run_triton_only 3 5
#   ./run_4_benchmarks.sh run_lif_triton [GPU_ID]     # e.g. ./run_4_benchmarks.sh run_lif_triton 3
#   ./run_4_benchmarks.sh run_cgrad_triton [GPU_ID]   # e.g. ./run_4_benchmarks.sh run_cgrad_triton 5
#   ./run_4_benchmarks.sh run_lif_pytorch [GPU_ID]
#   ./run_4_benchmarks.sh run_cgrad_pytorch [GPU_ID]
#
# (Also works when executed from repository root as: bash event/run_4_benchmarks.sh ...)
# ==============================================================================

# Ensure working directory is the event/ directory regardless of where script is invoked
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

set -e
MODE=${1:-sequential}
GPU_ID=${2:-4}

DATA_PATH="/data/rboone/datasets/wg_dvst"
CONFIG_FILE="./event_sweepcfile.yaml"
EPOCHS=96
BATCH_SIZE=16
T=16
SEED=44

# Common base arguments
BASE_ARGS="--config=${CONFIG_FILE} \
    --data-path=${DATA_PATH} \
    --dataset=cifar10dvs \
    --model=max_former \
    --T=${T} \
    --batch-size=${BATCH_SIZE} \
    --epochs=${EPOCHS} \
    --seed=${SEED} \
    --log-wandb"

run_lif_pytorch() {
    local target_gpu=${1:-$GPU_ID}
    echo "======================================================================"
    echo " [Run 1/4] Starting LIF (PyTorch Backend, Triton OFF) on GPU ${target_gpu}..."
    echo "======================================================================"
    CUDA_VISIBLE_DEVICES=${target_gpu} DISABLE_TRITON=1 conda run -n maxformer python train.py \
        ${BASE_ARGS} \
        --experiment="lif_pytorch_baseline" \
        --du-du="LIF" \
        --device="cuda:0"
}

run_lif_triton() {
    local target_gpu=${1:-$GPU_ID}
    echo "======================================================================"
    echo " [Run 2/4] Starting LIF (Triton Backend, Triton ON) on GPU ${target_gpu}..."
    echo "======================================================================"
    CUDA_VISIBLE_DEVICES=${target_gpu} DISABLE_TRITON=0 conda run -n maxformer python train.py \
        ${BASE_ARGS} \
        --experiment="lif_triton_accelerated" \
        --du-du="LIF" \
        --device="cuda:0"
}

run_cgrad_pytorch() {
    local target_gpu=${1:-$GPU_ID}
    echo "======================================================================"
    echo " [Run 3/4] Starting CGrad (PyTorch Backend, Triton OFF) on GPU ${target_gpu}..."
    echo "======================================================================"
    CUDA_VISIBLE_DEVICES=${target_gpu} DISABLE_TRITON=1 conda run -n maxformer python train.py \
        ${BASE_ARGS} \
        --experiment="cgrad_pytorch_reference" \
        --du-du="conservative_cgrad" \
        --device="cuda:0"
}

run_cgrad_triton() {
    local target_gpu=${1:-$GPU_ID}
    echo "======================================================================"
    echo " [Run 4/4] Starting CGrad (Triton Backend, Triton ON) on GPU ${target_gpu}..."
    echo "======================================================================"
    CUDA_VISIBLE_DEVICES=${target_gpu} DISABLE_TRITON=0 conda run -n maxformer python train.py \
        ${BASE_ARGS} \
        --experiment="cgrad_triton_accelerated" \
        --du-du="conservative_cgrad" \
        --device="cuda:0"
}

case "$MODE" in
    sequential)
        echo "Running all 4 training runs sequentially on GPU ${GPU_ID}..."
        run_lif_pytorch ${GPU_ID}
        run_lif_triton ${GPU_ID}
        run_cgrad_pytorch ${GPU_ID}
        run_cgrad_triton ${GPU_ID}
        ;;
    parallel)
        GPU_START=${GPU_ID}
        echo "Running all 4 training runs in parallel across GPUs $((GPU_START)), $((GPU_START+1)), $((GPU_START+2)), $((GPU_START+3))..."
        run_lif_pytorch $((GPU_START))   > logs/run_lif_pytorch.log 2>&1 &
        PID1=$!
        run_lif_triton $((GPU_START+1)) > logs/run_lif_triton.log 2>&1 &
        PID2=$!
        run_cgrad_pytorch $((GPU_START+2)) > logs/run_cgrad_pytorch.log 2>&1 &
        PID3=$!
        run_cgrad_triton $((GPU_START+3)) > logs/run_cgrad_triton.log 2>&1 &
        PID4=$!
        echo "Launched: PID1=${PID1} (GPU $((GPU_START))), PID2=${PID2} (GPU $((GPU_START+1))), PID3=${PID3} (GPU $((GPU_START+2))), PID4=${PID4} (GPU $((GPU_START+3)))"
        echo "Logs: logs/run_*.log"
        wait $PID1 $PID2 $PID3 $PID4
        echo "All 4 parallel runs completed!"
        ;;
    run_triton_only)
        GPU_A=${2:-3}
        GPU_B=${3:-5}
        echo "Running both Triton runs in parallel on GPU ${GPU_A} and GPU ${GPU_B}..."
        run_lif_triton ${GPU_A}   > logs/run_lif_triton.log 2>&1 &
        PID_L=$!
        run_cgrad_triton ${GPU_B} > logs/run_cgrad_triton.log 2>&1 &
        PID_C=$!
        echo "Launched Triton runs: LIF PID=${PID_L} (GPU ${GPU_A}), CGrad PID=${PID_C} (GPU ${GPU_B})"
        echo "Logs: logs/run_lif_triton.log and logs/run_cgrad_triton.log"
        wait $PID_L $PID_C
        echo "Both Triton runs completed!"
        ;;
    run_lif_pytorch)
        run_lif_pytorch $GPU_ID
        ;;
    run_lif_triton)
        run_lif_triton $GPU_ID
        ;;
    run_cgrad_pytorch)
        run_cgrad_pytorch $GPU_ID
        ;;
    run_cgrad_triton)
        run_cgrad_triton $GPU_ID
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: sequential | parallel | run_triton_only | run_lif_pytorch | run_lif_triton | run_cgrad_pytorch | run_cgrad_triton"
        exit 1
        ;;
esac
