#!/bin/bash
# Quick test: run a single vLLM annotation shard on blackwell1.
#
# Usage (from SNAP login node):
#   ssh ilc
#   cd /sailhome/moritzs/cellwhisperer/src/datasets/lbcl_atlas  # or wherever synced
#   sbatch test_single_shard.sh
#
# Or interactively:
#   srun --account=infolab --partition=il-interactive --gres=gpu:b200:1 --cpus-per-task=8 --mem=100G --time=02:00:00 bash test_single_shard.sh

#SBATCH --account=infolab
#SBATCH --partition=il-interactive
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --job-name=lbcl_test
#SBATCH --output=test_vllm_%j.out
#SBATCH --error=test_vllm_%j.err

# UV environment on local SSD
export UV_PROJECT_ENVIRONMENT=/lfs/local/0/$USER/uv-envs/lbcl_atlas
export XDG_CACHE_HOME=/lfs/local/0/$USER/.cache
export XDG_BIN_HOME=/lfs/local/0/$USER/.local/bin
export XDG_DATA_HOME=/lfs/local/0/$USER/.local/share

# Use SLURM_SUBMIT_DIR (the directory where sbatch was called)
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== LBCL Atlas vLLM test ==="
echo "Script dir: $SCRIPT_DIR"
echo "Project dir: $PROJECT_DIR"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "UV env: $UV_PROJECT_ENVIRONMENT"
date

cd "$SCRIPT_DIR"

# Create a small test input (3 samples) if no split YAML exists yet
TEST_YAML="$PROJECT_DIR/results/lbcl_atlas/requests/test_shard.yaml"
if [ ! -f "$TEST_YAML" ]; then
    echo "Creating test input YAML with 3 dummy samples..."
    mkdir -p "$(dirname "$TEST_YAML")"
    cat > "$TEST_YAML" << 'YAML'
lbcl_pseudobulk_0:
  cell_type: CD4-positive, alpha-beta T cell
  disease_state: B-cell non-Hodgkin lymphoma
  tissue: lymph node
  sex: male
  development_stage: 65-year-old human stage
  self_reported_ethnicity: European
  suspension_type: nucleus
  study_description: >-
    Large B-cell lymphomas (LBCL) are clinically and biologically diverse lymphoid
    malignancies with intricate microenvironments. Single-nucleus multiome profiling
    on 232 tumor and control biopsies.
  dataset_title: Single cell atlas of large B-cell lymphoma
lbcl_pseudobulk_1:
  cell_type: macrophage
  disease_state: normal
  tissue: lymph node
  sex: female
  development_stage: 45-year-old human stage
  self_reported_ethnicity: African American
  suspension_type: nucleus
  study_description: >-
    Large B-cell lymphomas (LBCL) are clinically and biologically diverse lymphoid
    malignancies with intricate microenvironments. Single-nucleus multiome profiling
    on 232 tumor and control biopsies.
  dataset_title: Single cell atlas of large B-cell lymphoma
lbcl_pseudobulk_2:
  cell_type: T follicular helper cell
  disease_state: B-cell non-Hodgkin lymphoma
  tissue: lymph node
  sex: male
  development_stage: 72-year-old human stage
  self_reported_ethnicity: European
  suspension_type: nucleus
  study_description: >-
    Large B-cell lymphomas (LBCL) are clinically and biologically diverse lymphoid
    malignancies with intricate microenvironments. Single-nucleus multiome profiling
    on 232 tumor and control biopsies.
  dataset_title: Single cell atlas of large B-cell lymphoma
YAML
    echo "Created $TEST_YAML"
fi

OUTPUT_CSV="$PROJECT_DIR/results/lbcl_atlas/processed/test_shard.csv"
LOG_FILE="$PROJECT_DIR/results/lbcl_atlas/logs/test_shard.log"
FEW_SHOT_DIR="$PROJECT_DIR/src/pre_training_processing/prompts/few_shot_samples"
PROMPT_FILE="$SCRIPT_DIR/prompts/annotation.txt"

echo "Running vLLM annotation..."
echo "Input: $TEST_YAML"
echo "Output: $OUTPUT_CSV"
echo "Prompt: $PROMPT_FILE"
echo "Few-shot dir: $FEW_SHOT_DIR"

export PYTORCH_ALLOC_CONF=expandable_segments:True

uv run --no-progress python scripts/run_vllm_annotation.py \
    --split-yaml "$TEST_YAML" \
    --prompt-file "$PROMPT_FILE" \
    --few-shot-dir "$FEW_SHOT_DIR" \
    --output-csv "$OUTPUT_CSV" \
    --log-file "$LOG_FILE" \
    --model openai/gpt-oss-120b \
    --max-num-seqs 16 \
    --tensor-parallel-size 1

echo ""
echo "=== Done ==="
echo "Output:"
cat "$OUTPUT_CSV" 2>/dev/null || echo "(no output)"
date
