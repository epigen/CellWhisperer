#!/bin/bash
# Launch UCE training experiment on Sherlock.
#
# Tests the new UCE model (KuanP/uce-cosmx-geneset) as transcriptome encoder
# vs the baseline MLP encoder. Assumes UCE caches are already preprocessed.
#
# Usage:
#   bash launch_uce.sh              # submit training job
#   bash launch_uce.sh --preprocess # submit CPU-only preprocessing job
#   bash launch_uce.sh --dry-run    # print what would be submitted

set -eo pipefail  # no -u: conda activate uses unbound vars

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="train"
if [[ "${1:-}" == "--dry-run" ]]; then
    echo "=== DRY RUN — no jobs will be submitted ==="
    MODE="dry-run"
elif [[ "${1:-}" == "--preprocess" ]]; then
    MODE="preprocess"
fi

# Common SLURM settings
PARTITION="cmackall"
ACCOUNT="zinaida"
LOG_DIR="/scratch/users/moritzs/lymphoma_trimodal_opt_logs"
mkdir -p "$LOG_DIR"

NAME="uce"
DELTA="delta_config/uce.yaml"

if [[ "$MODE" == "preprocess" ]]; then
    # CPU-only preprocessing: extract UNI patches + UCE embeddings
    # Needs lots of RAM (datasets are large), no GPU required
    SBATCH_ARGS=(
        --cpus-per-task=16
        --mem=600G
        --time=2-00:00:00
    )
    JOB_CMD="cellwhisperer fit --config base_config.yaml --config $DELTA --trainer.fast_dev_run=true --trainer.logger=false --trainer.accelerator=cpu || true"
else
    # GPU training: data is cached on disk, loaded on-the-fly
    SBATCH_ARGS=(
        -G 1
        --cpus-per-task=8
        --mem=200G
        --time=2-00:00:00
    )
    JOB_CMD="cellwhisperer fit --config base_config.yaml --config $DELTA"
fi

# Build the SLURM inline script (source activate, not conda run, for unbuffered output)
WRAP_SCRIPT="set -eo pipefail; cd ${SCRIPT_DIR}; source activate cellwhisperer; export PYTHONUNBUFFERED=1; ${JOB_CMD}"

echo "Launching UCE ${MODE} job:"
echo "  Config: base_config.yaml + $DELTA"
echo "  Command: $JOB_CMD"
echo "  Logs: ${LOG_DIR}/${NAME}-${MODE}_*.{out,err}"
echo ""

if [[ "$MODE" == "dry-run" ]]; then
    echo "Would run: sbatch --account=$ACCOUNT --partition=$PARTITION ${SBATCH_ARGS[*]} --wrap='...'"
    exit 0
fi

sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    "${SBATCH_ARGS[@]}" \
    --job-name="trimodal-${NAME}-${MODE}" \
    --output="${LOG_DIR}/${NAME}-${MODE}_%j.out" \
    --error="${LOG_DIR}/${NAME}-${MODE}_%j.err" \
    --wrap="$WRAP_SCRIPT"

echo "Job submitted. Monitor with: squeue -u \$USER"
echo "View logs with: tail -f ${LOG_DIR}/${NAME}-${MODE}_*.out"
