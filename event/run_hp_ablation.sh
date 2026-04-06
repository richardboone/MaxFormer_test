#!/bin/bash
# =============================================================================
# Hyperparameter Ablation Study: conservative_cgrad on CIFAR10-DVS
# One-at-a-time ablation: each HP gets one run ABOVE default and one BELOW.
# 1 baseline + 5 HPs × 2 levels = 11 total runs.
# Batches across GPUs, waits for each batch to finish before starting next.
# =============================================================================

# --- Configuration ---
GPUS=(0 1 2)
NUM_GPUS=${#GPUS[@]}
DATA_PATH="/data/rboone/datasets/wg_dvst"
CONFIG="./cifar10dvs_hp_ablation.yaml"
DU_DU="conservative_cgrad"
DS_DU="Gamma"
DRY_RUN=${DRY_RUN:-0}   # Set DRY_RUN=1 to print commands without executing

# --- Default hyperparameter values ---
DEF_ALPHA_M=1.2
DEF_ALPHA_D=1.3
DEF_H=0.5
DEF_B=0.5
DEF_ETA=0.5

# --- Ablation levels (low, high) for each HP ---
ALPHA_M_LO=0.8;   ALPHA_M_HI=2.0
ALPHA_D_LO=0.8;   ALPHA_D_HI=2.0
H_LO=0.3;         H_HI=0.8
B_LO=0.3;         B_HI=0.8
ETA_LO=0.3;       ETA_HI=0.8

# ---------------------------------------------------------------
# Helper: build a single training command
# Arguments: GPU_ID  EXP_NAME  ALPHA_M  ALPHA_D  H  B  ETA
# ---------------------------------------------------------------
build_cmd() {
  local exp_name=$1 alpha_m=$2 alpha_d=$3 h=$4 b=$5 eta=$6
  echo "python train.py \
-c ${CONFIG} \
--data-path ${DATA_PATH} \
--log-wandb \
--model max_former \
--dS-du ${DS_DU} \
--du-du ${DU_DU} \
--snnbp-alpha ${alpha_m} \
--snnbp-beta ${alpha_d} \
--snnbp-intervention ${h} \
--snnbp-blend ${b} \
--snnbp-eta ${eta} \
--experiment ${exp_name}"
}

# --- Build all commands ---
COMMANDS=()
NAMES=()

# 0) Baseline (all defaults)
NAME="hp_ablate_baseline"
COMMANDS+=("$(build_cmd $NAME $DEF_ALPHA_M $DEF_ALPHA_D $DEF_H $DEF_B $DEF_ETA)")
NAMES+=("$NAME")

# 1) alpha_m low / high
NAME="hp_ablate_alpha_m_${ALPHA_M_LO}"
COMMANDS+=("$(build_cmd $NAME $ALPHA_M_LO $DEF_ALPHA_D $DEF_H $DEF_B $DEF_ETA)")
NAMES+=("$NAME")
NAME="hp_ablate_alpha_m_${ALPHA_M_HI}"
COMMANDS+=("$(build_cmd $NAME $ALPHA_M_HI $DEF_ALPHA_D $DEF_H $DEF_B $DEF_ETA)")
NAMES+=("$NAME")

# 2) alpha_d low / high
NAME="hp_ablate_alpha_d_${ALPHA_D_LO}"
COMMANDS+=("$(build_cmd $NAME $DEF_ALPHA_M $ALPHA_D_LO $DEF_H $DEF_B $DEF_ETA)")
NAMES+=("$NAME")
NAME="hp_ablate_alpha_d_${ALPHA_D_HI}"
COMMANDS+=("$(build_cmd $NAME $DEF_ALPHA_M $ALPHA_D_HI $DEF_H $DEF_B $DEF_ETA)")
NAMES+=("$NAME")

# 3) h (intervention threshold) low / high
NAME="hp_ablate_h_${H_LO}"
COMMANDS+=("$(build_cmd $NAME $DEF_ALPHA_M $DEF_ALPHA_D $H_LO $DEF_B $DEF_ETA)")
NAMES+=("$NAME")
NAME="hp_ablate_h_${H_HI}"
COMMANDS+=("$(build_cmd $NAME $DEF_ALPHA_M $DEF_ALPHA_D $H_HI $DEF_B $DEF_ETA)")
NAMES+=("$NAME")

# 4) b (blend scale) low / high
NAME="hp_ablate_b_${B_LO}"
COMMANDS+=("$(build_cmd $NAME $DEF_ALPHA_M $DEF_ALPHA_D $DEF_H $B_LO $DEF_ETA)")
NAMES+=("$NAME")
NAME="hp_ablate_b_${B_HI}"
COMMANDS+=("$(build_cmd $NAME $DEF_ALPHA_M $DEF_ALPHA_D $DEF_H $B_HI $DEF_ETA)")
NAMES+=("$NAME")

# 5) eta (correction magnitude) low / high
NAME="hp_ablate_eta_${ETA_LO}"
COMMANDS+=("$(build_cmd $NAME $DEF_ALPHA_M $DEF_ALPHA_D $DEF_H $DEF_B $ETA_LO)")
NAMES+=("$NAME")
NAME="hp_ablate_eta_${ETA_HI}"
COMMANDS+=("$(build_cmd $NAME $DEF_ALPHA_M $DEF_ALPHA_D $DEF_H $DEF_B $ETA_HI)")
NAMES+=("$NAME")

# --- Launch ---
TOTAL=${#COMMANDS[@]}
echo "========================================"
echo " HP Ablation: conservative_cgrad"
echo " Total runs: ${TOTAL}  (1 baseline + 5×2 ablations)"
echo " GPUs: ${GPUS[*]}"
echo "========================================"
echo ""

idx=0
while [ $idx -lt $TOTAL ]; do
  PIDS=()

  for g in $(seq 0 $((NUM_GPUS - 1))); do
    run_idx=$((idx + g))
    if [ $run_idx -ge $TOTAL ]; then
      break
    fi

    GPU=${GPUS[$g]}
    EXP_NAME=${NAMES[$run_idx]}
    CMD=${COMMANDS[$run_idx]}

    echo "=== Run $((run_idx + 1))/${TOTAL}: ${EXP_NAME} on GPU ${GPU} ==="

    if [ "$DRY_RUN" = "1" ]; then
      echo "CUDA_VISIBLE_DEVICES=${GPU} ${CMD}"
      echo ""
    else
      CUDA_VISIBLE_DEVICES=${GPU} ${CMD} &
      PIDS+=($!)
    fi
  done

  # Wait for this batch to finish
  if [ "$DRY_RUN" != "1" ] && [ ${#PIDS[@]} -gt 0 ]; then
    echo "--- Waiting for batch (runs $((idx + 1))-$((idx + ${#PIDS[@]}))) ---"
    for pid in "${PIDS[@]}"; do
      wait $pid
    done
    echo "--- Batch complete ---"
    echo ""
  fi

  idx=$((idx + NUM_GPUS))
done

echo "All ${TOTAL} ablation runs complete."
