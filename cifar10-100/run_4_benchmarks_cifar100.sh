#!/bin/bash
# ==============================================================================
# 4 MaxFormer Timing & Parity Benchmark Runs on CIFAR-100
#
# Matrix of 4 configurations:
#   1. LIF   + PyTorch Autograd (Triton OFF)
#   2. LIF   + Fused Triton GPU Kernels (Triton ON)
#   3. CGrad + PyTorch Autograd (Triton OFF)
#   4. CGrad + Fused Triton GPU Kernels (Triton ON)
#
# Usage (from inside cifar10-100/ or repo root):
#   ./run_4_benchmarks_cifar100.sh sequential [GPU_ID]
#   ./run_4_benchmarks_cifar100.sh parallel [GPU0] [GPU1] [GPU2] [GPU3]
#   ./run_4_benchmarks_cifar100.sh run_triton_only [GPU1] [GPU2]
#   ./run_4_benchmarks_cifar100.sh run_lif_pytorch [GPU_ID]
#   ./run_4_benchmarks_cifar100.sh run_lif_triton [GPU_ID]
#   ./run_4_benchmarks_cifar100.sh run_cgrad_pytorch [GPU_ID]
#   ./run_4_benchmarks_cifar100.sh run_cgrad_triton [GPU_ID]
#
# Environment overrides:
#   EPOCHS=10        (default: 10; set EPOCHS=400 for full training)
#   TIME_STEP=8      (default: 8)
#   BATCH_SIZE=64    (default: 64)
#   LOG_WANDB=0      (default: 0; set LOG_WANDB=1 to enable W&B logging)
#   DATA_PATH=/data/rboone/datasets/cifar100/
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

set -e
MODE=${1:-sequential}
GPU_ID=${2:-0}

DATA_PATH="${DATA_PATH:-/data/rboone/datasets/cifar100/}"
CONFIG_FILE="${CONFIG_FILE:-./cifar100_best_cons.yaml}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TIME_STEP="${TIME_STEP:-8}"
SEED="${SEED:-50425}"
LOG_WANDB="${LOG_WANDB:-0}"

WANDB_FLAG=""
if [ "${LOG_WANDB}" = "1" ]; then
    WANDB_FLAG="--log-wandb"
fi

BASE_ARGS="--config=${CONFIG_FILE} \
    --data-path=${DATA_PATH} \
    --dataset=torch/cifar100 \
    --model=max_former \
    --time-step=${TIME_STEP} \
    --batch-size=${BATCH_SIZE} \
    --epochs=${EPOCHS} \
    --seed=${SEED} \
    --use-custom-neuron=True \
    --dS-du=Gamma \
    ${WANDB_FLAG}"

run_lif_pytorch() {
    local target_gpu=${1:-$GPU_ID}
    echo "======================================================================"
    echo " [Run 1/4] Starting CIFAR-100 LIF (PyTorch, Triton OFF) on GPU ${target_gpu}..."
    echo "======================================================================"
    CUDA_VISIBLE_DEVICES=${target_gpu} DISABLE_TRITON=1 conda run --no-capture-output -n maxformer python train.py \
        ${BASE_ARGS} \
        --experiment="c100_lif_pytorch_baseline" \
        --du-du="LIF"
}

run_lif_triton() {
    local target_gpu=${1:-$GPU_ID}
    echo "======================================================================"
    echo " [Run 2/4] Starting CIFAR-100 LIF (Triton ON) on GPU ${target_gpu}..."
    echo "======================================================================"
    CUDA_VISIBLE_DEVICES=${target_gpu} DISABLE_TRITON=0 conda run --no-capture-output -n maxformer python train.py \
        ${BASE_ARGS} \
        --experiment="c100_lif_triton_accelerated" \
        --du-du="LIF"
}

run_cgrad_pytorch() {
    local target_gpu=${1:-$GPU_ID}
    echo "======================================================================"
    echo " [Run 3/4] Starting CIFAR-100 CGrad (PyTorch, Triton OFF) on GPU ${target_gpu}..."
    echo "======================================================================"
    CUDA_VISIBLE_DEVICES=${target_gpu} DISABLE_TRITON=1 conda run --no-capture-output -n maxformer python train.py \
        ${BASE_ARGS} \
        --experiment="c100_cgrad_pytorch_reference" \
        --du-du="conservative_cgrad"
}

run_cgrad_triton() {
    local target_gpu=${1:-$GPU_ID}
    echo "======================================================================"
    echo " [Run 4/4] Starting CIFAR-100 CGrad (Triton ON) on GPU ${target_gpu}..."
    echo "======================================================================"
    CUDA_VISIBLE_DEVICES=${target_gpu} DISABLE_TRITON=0 conda run --no-capture-output -n maxformer python train.py \
        ${BASE_ARGS} \
        --experiment="c100_cgrad_triton_accelerated" \
        --du-du="conservative_cgrad"
}

case "$MODE" in
    sequential)
        echo "Running all 4 CIFAR-100 runs sequentially on GPU ${GPU_ID} (EPOCHS=${EPOCHS}, T=${TIME_STEP})..."
        run_lif_pytorch "${GPU_ID}"   2>&1 | tee logs/run_c100_lif_pytorch.log
        run_lif_triton "${GPU_ID}"    2>&1 | tee logs/run_c100_lif_triton.log
        run_cgrad_pytorch "${GPU_ID}" 2>&1 | tee logs/run_c100_cgrad_pytorch.log
        run_cgrad_triton "${GPU_ID}"  2>&1 | tee logs/run_c100_cgrad_triton.log
        ;;
    parallel)
        if [ -n "$3" ] && [ -n "$4" ] && [ -n "$5" ]; then
            GPU0=$2; GPU1=$3; GPU2=$4; GPU3=$5
        else
            GPU0=${GPU_ID}; GPU1=$((GPU_ID+1)); GPU2=$((GPU_ID+2)); GPU3=$((GPU_ID+3))
        fi
        echo "Running all 4 CIFAR-100 runs in parallel on GPUs ${GPU0}, ${GPU1}, ${GPU2}, ${GPU3} (EPOCHS=${EPOCHS}, T=${TIME_STEP})..."
        run_lif_pytorch "${GPU0}"   > logs/run_c100_lif_pytorch.log 2>&1 &
        PID1=$!
        run_lif_triton "${GPU1}"    > logs/run_c100_lif_triton.log 2>&1 &
        PID2=$!
        run_cgrad_pytorch "${GPU2}" > logs/run_c100_cgrad_pytorch.log 2>&1 &
        PID3=$!
        run_cgrad_triton "${GPU3}"  > logs/run_c100_cgrad_triton.log 2>&1 &
        PID4=$!
        echo "Launched: PID1=${PID1} (GPU ${GPU0}), PID2=${PID2} (GPU ${GPU1}), PID3=${PID3} (GPU ${GPU2}), PID4=${PID4} (GPU ${GPU3})"
        echo "Logs: logs/run_c100_*.log"
        wait $PID1 $PID2 $PID3 $PID4
        echo "All 4 parallel CIFAR-100 runs completed!"
        ;;
    run_triton_only)
        GPU_A=${2:-0}
        GPU_B=${3:-1}
        echo "Running both CIFAR-100 Triton runs in parallel on GPU ${GPU_A} and GPU ${GPU_B}..."
        run_lif_triton "${GPU_A}"   > logs/run_c100_lif_triton.log 2>&1 &
        PID_L=$!
        run_cgrad_triton "${GPU_B}" > logs/run_c100_cgrad_triton.log 2>&1 &
        PID_C=$!
        echo "Launched Triton runs: LIF PID=${PID_L} (GPU ${GPU_A}), CGrad PID=${PID_C} (GPU ${GPU_B})"
        echo "Logs: logs/run_c100_lif_triton.log and logs/run_c100_cgrad_triton.log"
        wait $PID_L $PID_C
        echo "Both Triton runs completed!"
        ;;
    run_lif_pytorch)
        run_lif_pytorch "$GPU_ID"
        ;;
    run_lif_triton)
        run_lif_triton "$GPU_ID"
        ;;
    run_cgrad_pytorch)
        run_cgrad_pytorch "$GPU_ID"
        ;;
    run_cgrad_triton)
        run_cgrad_triton "$GPU_ID"
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: sequential | parallel | run_triton_only | run_lif_pytorch | run_lif_triton | run_cgrad_pytorch | run_cgrad_triton"
        exit 1
        ;;
esac
