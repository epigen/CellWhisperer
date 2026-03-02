#!/bin/bash
# Run the full core-level analysis pipeline for TMA2 on Sherlock.
#
# This script should be submitted as a SLURM job from a compute node or login node:
#   sbatch --account=zinaida --partition=cmackall -G 1 --cpus-per-task=10 --mem=80G --time=04:00:00 \
#       --output=$HOME/scratch/logs/core_level_tma2_%j.out \
#       --error=$HOME/scratch/logs/core_level_tma2_%j.err \
#       scripts/run_core_level_tma2.sh

set -euo pipefail

# Navigate to project root
cd ~/cellwhisperer_private

echo "=== Core-Level TMA2 Analysis Pipeline ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'No GPU')"
echo ""

RESULTS_DIR="results/experiments/932-tma1-ihc-eval"
SCRIPTS_DIR="src/experiments/932-tma1-ihc-eval/scripts"
CHECKPOINT="results/models/spatialwhisperer/v1.ckpt"
CORE_DIR="${RESULTS_DIR}/metrics/core_level"
PLOTS_DIR="${RESULTS_DIR}/plots/core_level_correlation"

mkdir -p "$CORE_DIR" "$PLOTS_DIR"

# Step 1: Compute core-level decoder performance (CPU-only, fast)
echo "=== Step 1/3: Core-level decoder performance ==="
conda run -n cellwhisperer python "${SCRIPTS_DIR}/compute_core_decoder_performance.py" \
    --predictions "${RESULTS_DIR}/predictions/TMA2_predictions.h5ad" \
    --top_n 500 \
    --min_cells 50 \
    --output "${CORE_DIR}/TMA2_core_decoder.csv"
echo "Done."
echo ""

# Step 2: Compute core-level base model metrics (needs GPU for embedding extraction)
echo "=== Step 2/3: Core-level retrieval + loss ==="
conda run -n cellwhisperer python "${SCRIPTS_DIR}/compute_core_base_metrics.py" \
    --checkpoint "$CHECKPOINT" \
    --dataset_name lymphoma_cosmx_large_TMA2 \
    --predictions "${RESULTS_DIR}/predictions/TMA2_predictions.h5ad" \
    --batch_size 64 \
    --nproc 8 \
    --min_cells 50 \
    --output_retrieval "${CORE_DIR}/TMA2_core_retrieval.csv" \
    --output_loss "${CORE_DIR}/TMA2_core_loss.csv"
echo "Done."
echo ""

# Step 3: Correlate metrics (CPU-only, fast)
echo "=== Step 3/3: Correlation analysis ==="
conda run -n cellwhisperer python "${SCRIPTS_DIR}/correlate_core_metrics.py" \
    --retrieval_csvs "${CORE_DIR}/TMA2_core_retrieval.csv" \
    --loss_csvs "${CORE_DIR}/TMA2_core_loss.csv" \
    --decoder_csvs "${CORE_DIR}/TMA2_core_decoder.csv" \
    --output_combined "${CORE_DIR}/core_metrics_combined.csv" \
    --output_corr_matrix "${CORE_DIR}/correlation_matrix.csv" \
    --output_plots "$PLOTS_DIR"
echo "Done."
echo ""

echo "=== Pipeline complete ==="
echo "Results in: ${CORE_DIR}/"
echo "Plots in: ${PLOTS_DIR}/"
echo "Date: $(date)"
