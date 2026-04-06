#!/bin/bash
# Run all best configurations sequentially for comprehensive evaluation
# This script runs both configurations with unique experiment names for easy comparison in wandb

echo "=========================================="
echo "Starting evaluation of best sweep configs"
echo "=========================================="
echo ""

# Configuration 1
echo "[1/2] Running Configuration 1..."
echo "Characteristics: Higher alpha/beta (2.0/3.6), higher weight decay (0.02)"
bash run_config1.sh
CONFIG1_STATUS=$?

if [ $CONFIG1_STATUS -ne 0 ]; then
    echo "WARNING: Configuration 1 failed with exit code $CONFIG1_STATUS"
fi

echo ""
echo "=========================================="
echo ""

# Configuration 2
echo "[2/2] Running Configuration 2..."
echo "Characteristics: Lower alpha/beta (1.3/1.3), higher k-dir (2.9)"
bash run_config2.sh
CONFIG2_STATUS=$?

if [ $CONFIG2_STATUS -ne 0 ]; then
    echo "WARNING: Configuration 2 failed with exit code $CONFIG2_STATUS"
fi

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Check wandb for results: experiment names 'event_config1_eval' and 'event_config2_eval'"
echo "=========================================="
