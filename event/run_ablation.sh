#!/bin/bash
# =============================================================================
# Ablation Study: conservative_ablate on CIFAR10-DVS
# Runs all 32 combinations of 5 binary ablation flags.
# 3 runs in parallel at once (one per GPU), waits for all 3 to finish,
# then launches the next batch of 3.
# =============================================================================

# --- Configuration ---
GPUS=(0 1 2)
NUM_GPUS=${#GPUS[@]}
DATA_PATH="/data/rboone/datasets/wg_dvst"
CONFIG="./cifar10dvs_rebuttal_ablate.yaml"
DU_DU="conservative_ablate"
DS_DU="Gamma"
DRY_RUN=${DRY_RUN:-0}  # Set DRY_RUN=1 to print commands without executing
DETACH="true"

# --- Build all 32 combinations into an array ---
COMMANDS=()
NAMES=()

for gm in true false; do
for gd in true false; do
    for gmisalign in true false; do
    for intervention in true false; do

        # Build short tag for experiment name
        d_tag=$( [ "$DETACH" = "true" ] && echo "D1" || echo "D0" )
        gm_tag=$( [ "$gm" = "true" ] && echo "gm1" || echo "gm0" )
        gd_tag=$( [ "$gd" = "true" ] && echo "gd1" || echo "gd0" )
        ga_tag=$( [ "$gmisalign" = "true" ] && echo "ga1" || echo "ga0" )
        it_tag=$( [ "$intervention" = "true" ] && echo "it1" || echo "it0" )

        EXP_NAME="ablate_opti_${d_tag}_${gm_tag}_${gd_tag}_${ga_tag}_${it_tag}"

        CMD="python train.py \
-c ${CONFIG} \
--data-path ${DATA_PATH} \
--log-wandb \
--model max_former \
--dS-du ${DS_DU} \
--du-du ${DU_DU} \
--detach-reset ${DETACH} \
--ablation-gm ${gm} \
--ablation-gd ${gd} \
--ablation-gmisalign ${gmisalign} \
--ablation-intervention ${intervention} \
--experiment ${EXP_NAME}"

        COMMANDS+=("$CMD")
        NAMES+=("$EXP_NAME")

    done
    done
done
done

TOTAL=${#COMMANDS[@]}
echo "Total ablation runs: ${TOTAL}"
echo "GPUs: ${GPUS[*]}"
echo ""

# --- Launch in batches of NUM_GPUS ---
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

    echo "=== Run ${run_idx}/${TOTAL}: ${EXP_NAME} on GPU ${GPU} ==="

    if [ "$DRY_RUN" = "1" ]; then
      echo "CUDA_VISIBLE_DEVICES=${GPU} ${CMD}"
      echo ""
    else
      CUDA_VISIBLE_DEVICES=${GPU} ${CMD} &
      PIDS+=($!)
    fi
  done

  # Wait for this batch to finish before starting the next
  if [ "$DRY_RUN" != "1" ] && [ ${#PIDS[@]} -gt 0 ]; then
    echo "--- Waiting for batch (runs ${idx}-$((idx + ${#PIDS[@]} - 1))) to finish ---"
    for pid in "${PIDS[@]}"; do
      wait $pid
    done
    echo "--- Batch complete ---"
    echo ""
  fi

  idx=$((idx + NUM_GPUS))
done

echo "All ${TOTAL} ablation runs complete."
