#!/bin/bash
# Launch all lymphoma trimodal optimization experiments on Sherlock.
#
# Each experiment runs as a separate SLURM job with 1 GPU.
# Usage: bash launch_all.sh [--dry-run]
#
# Experiments (9 total):
#   1. baseline           — base_config.yaml only
#   2. census_downweight  — reduce census weight
#   3. larger_cnn         — bigger CNN
#   4. hqcores_train      — filtered train + filtered eval
#   5. hqcores_ref        — unfiltered train + filtered eval
#   6. nofna_train        — no-FNA train + no-FNA eval
#   7. nofna_ref          — unfiltered train + no-FNA eval
#   8. raw_train          — raw counts train + raw eval
#   9. raw_ref            — SCT counts train + raw eval

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — no jobs will be submitted ==="
fi

# SLURM settings
PARTITION="cmackall"
ACCOUNT="zinaida"
GPUS=1
CPUS=8
MEM="200G"
TIME="12:00:00"
LOG_DIR="/scratch/users/moritzs/lymphoma_trimodal_opt_logs"

mkdir -p "$LOG_DIR"

# All delta configs to run (empty string = baseline with base_config only)
declare -A EXPERIMENTS
EXPERIMENTS=(
    ["baseline"]=""
    ["census_downweight"]="delta_config/census_downweight.yaml"
    ["larger_cnn"]="delta_config/larger_cnn.yaml"
    ["hqcores_train"]="delta_config/hqcores_train.yaml"
    ["hqcores_ref"]="delta_config/hqcores_ref.yaml"
    ["nofna_train"]="delta_config/nofna_train.yaml"
    ["nofna_ref"]="delta_config/nofna_ref.yaml"
    ["raw_train"]="delta_config/raw_train.yaml"

)

echo "Submitting ${#EXPERIMENTS[@]} experiments from: $SCRIPT_DIR"
echo ""

for name in "${!EXPERIMENTS[@]}"; do
    delta="${EXPERIMENTS[$name]}"

    # Build cellwhisperer command
    CMD="cellwhisperer fit --config base_config.yaml"
    if [[ -n "$delta" ]]; then
        CMD="$CMD --config $delta"
    fi

    echo "[$name] $CMD"

    if $DRY_RUN; then
        continue
    fi

    sbatch \
        --account="$ACCOUNT" \
        --partition="$PARTITION" \
        -G "$GPUS" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        --job-name="trimodal-${name}" \
        --output="${LOG_DIR}/${name}_%j.out" \
        --error="${LOG_DIR}/${name}_%j.err" \
        --wrap="cd ${SCRIPT_DIR} && conda run -n cellwhisperer ${CMD}"

done

echo ""
echo "Done. Logs will be in: $LOG_DIR"
echo "Monitor with: squeue -u \$USER"
