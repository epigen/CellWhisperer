#!/bin/bash
# Launch UCE experiment on Sherlock to test new UCE implementation.
#
# This tests the new UCE model (KuanP/uce-cosmx-geneset) as transcriptome encoder
# vs the baseline MLP encoder.
#
# Usage: bash launch_uce.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — no jobs will be submitted ==="
fi

# SLURM settings (matching other experiments)
PARTITION="cmackall"
ACCOUNT="zinaida"
GPUS=1
CPUS=8
MEM="200G"
TIME="12:00:00"
LOG_DIR="/scratch/users/moritzs/lymphoma_trimodal_opt_logs"

mkdir -p "$LOG_DIR"

# UCE experiment
NAME="uce"
DELTA="delta_config/uce.yaml"
CMD="cellwhisperer fit --config base_config.yaml --config $DELTA"

echo "Launching UCE experiment:"
echo "  Config: base_config.yaml + $DELTA"
echo "  Command: $CMD"
echo "  Logs: ${LOG_DIR}/${NAME}_*.{out,err}"
echo ""

if $DRY_RUN; then
    echo "DRY RUN - would submit SLURM job with above settings"
    exit 0
fi

sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    -G "$GPUS" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --job-name="trimodal-${NAME}" \
    --output="${LOG_DIR}/${NAME}_%j.out" \
    --error="${LOG_DIR}/${NAME}_%j.err" \
    --wrap="cd ${SCRIPT_DIR} && conda run -n cellwhisperer ${CMD}"

echo "Job submitted. Monitor with: squeue -u \$USER"
echo "View logs with: tail -f ${LOG_DIR}/${NAME}_*.out"
