#!/bin/bash

# CosMx1K Evaluation Experiment (918) - SLURM Launch Script
# This script launches all experimental variants as separate SLURM jobs

# Dynamically determine project root and experiment directory
PROJECT_ROOT=$(git rev-parse --show-toplevel)
EXPERIMENT_DIR="$PROJECT_ROOT/src/experiments/918-cosmx1k-eval"

# Detect conda installation path via multiple methods
CONDA_PROFILE=""
CONDA_ROOT=""
CONDA_ENV=""

# Method 1: Use environment variables from server (if available)
if [ -n "$_CONDA_ROOT" ]; then
    # Use _CONDA_ROOT if available (most reliable on server)
    CONDA_ROOT="$_CONDA_ROOT"
    CONDA_PROFILE="$_CONDA_ROOT/etc/profile.d/conda.sh"
elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
    # Use CONDA_PREFIX if it contains conda.sh
    CONDA_ROOT="$CONDA_PREFIX"
    CONDA_PROFILE="$CONDA_PREFIX/etc/profile.d/conda.sh"
elif [ -n "$CONDA_EXE" ]; then
    # Derive conda root from CONDA_EXE path
    CONDA_ROOT="$(dirname "$(dirname "$CONDA_EXE")")"
    CONDA_PROFILE="$CONDA_ROOT/etc/profile.d/conda.sh"
fi

# Method 2: Fallback to common installation paths if env vars not available
if [ -z "$CONDA_PROFILE" ] || [ ! -f "$CONDA_PROFILE" ]; then
    # Try common conda installation paths
    for path in "$HOME/.mamba" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
        if [ -f "$path/etc/profile.d/conda.sh" ]; then
            CONDA_ROOT="$path"
            CONDA_PROFILE="$path/etc/profile.d/conda.sh"
            break
        fi
    done
fi

# Method 3: Try to find conda in PATH
if [ -z "$CONDA_PROFILE" ] || [ ! -f "$CONDA_PROFILE" ]; then
    conda_path=$(which conda 2>/dev/null)
    if [ -n "$conda_path" ]; then
        # Derive conda root from conda executable path
        CONDA_ROOT="$(dirname "$(dirname "$conda_path")")"
        CONDA_PROFILE="$CONDA_ROOT/etc/profile.d/conda.sh"
    fi
fi

# Verify conda.sh exists
if [ -z "$CONDA_PROFILE" ] || [ ! -f "$CONDA_PROFILE" ]; then
    echo "Error: Could not detect conda installation"
    echo "Tried the following methods:"
    echo "  1. Environment variables (_CONDA_ROOT, CONDA_PREFIX, CONDA_EXE)"
    echo "  2. Common paths (~/.mamba, ~/miniforge3, ~/miniconda3, ~/anaconda3)"
    echo "  3. PATH lookup"
    echo ""
    echo "Available conda env vars:"
    env | grep -i conda || echo "  (none found)"
    echo ""
    echo "Conda in PATH: $(which conda 2>/dev/null || echo "not found")"
    exit 1
fi

# Detect appropriate conda environment
# Check if we can access conda and if cellwhisperer variants exist
if [ -f "$CONDA_PROFILE" ]; then
    # Source conda to get access to conda command
    source "$CONDA_PROFILE" 2>/dev/null
    
    if command -v conda >/dev/null 2>&1; then
        # List available environments and choose the best cellwhisperer environment
        if conda env list 2>/dev/null | grep -q "cellwhisperer-aarch64"; then
            CONDA_ENV="cellwhisperer-aarch64"
        elif conda env list 2>/dev/null | grep -q "cellwhisperer"; then
            CONDA_ENV="cellwhisperer"
        else
            echo "Warning: No cellwhisperer environment found"
            echo "Available environments:"
            conda env list 2>/dev/null || echo "  (could not list environments)"
            echo "Defaulting to 'cellwhisperer'"
            CONDA_ENV="cellwhisperer"
        fi
    else
        # Fall back to common environment name
        CONDA_ENV="cellwhisperer"
    fi
else
    CONDA_ENV="cellwhisperer"
fi

# Array of delta config files
declare -a DELTA_CONFIGS=(
    "disable_cell_level_config.yaml"
    "finetune_geneformer_config.yaml" 
    "finetune_uni_config.yaml"
    "good_quality_cells_config.yaml"
    "nonaligned_cores_config.yaml"
    "ramp_up_cnn_config.yaml"
    "cell_cnn_only_config.yaml"
)

echo "Launching CosMx1K evaluation jobs..."
echo "Project root: $PROJECT_ROOT"
echo "Experiment directory: $EXPERIMENT_DIR"
echo "Conda root: $CONDA_ROOT"
echo "Conda profile: $CONDA_PROFILE"
echo "Conda environment: $CONDA_ENV"
echo "Number of delta configs: ${#DELTA_CONFIGS[@]}"
echo "Total jobs (including baseline): $((${#DELTA_CONFIGS[@]} + 1))"
echo

# Create logs directory if it doesn't exist
mkdir -p "$EXPERIMENT_DIR/slurm_logs"

# Function to create and submit a SLURM job
submit_job() {
    local job_name="$1"
    local config_args="$2"
    local description="$3"
    
    echo "Launching job: $job_name"
    echo "  Description: $description"
    echo "  Config args: $config_args"
    
    # Create a temporary sbatch script for this specific job
    sbatch_script="$EXPERIMENT_DIR/slurm_logs/sbatch_${job_name}.job"
    
    cat > "$sbatch_script" << EOF
#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --time=48:00:00
#SBATCH --partition=cmackall
#SBATCH --mem=150000
#SBATCH -G 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 12
#SBATCH -o $EXPERIMENT_DIR/slurm_logs/slurm-%j-${job_name}.out
#SBATCH -e $EXPERIMENT_DIR/slurm_logs/slurm-%j-${job_name}.err

# Source conda environment
source $CONDA_PROFILE
conda activate $CONDA_ENV

# Set SLURM environment variables
export GLOBAL_RANK=\$SLURM_LOCALID
export LOCAL_RANK=\$SLURM_LOCALID  
export RANK=\$SLURM_LOCALID

# Move to project directory for wandb git tracking
cd $PROJECT_ROOT

echo 'Starting job: $job_name'
echo 'Description: $description'
echo 'Working directory: \$(pwd)'
echo 'Time: \$(date)'
echo 'Job ID: \$SLURM_JOB_ID'
echo 'Node: \$SLURM_NODELIST'
echo

# Run the training command from experiment directory
cd $EXPERIMENT_DIR
cellwhisperer fit $config_args

echo
echo 'Job completed: $job_name'
echo 'End time: \$(date)'
EOF
    
    # Submit the job
    jobid=$(sbatch "$sbatch_script" | awk '{print $4}')
    
    echo "  Job submitted with ID: $jobid"
    echo "  Log file: slurm_logs/slurm-${jobid}-${job_name}.out"
    echo
    
    # Small delay to avoid overwhelming the scheduler
    sleep 2
}

# Launch baseline job (base config only)
submit_job "cosmx1k-baseline" "--config base_config.yaml" "Baseline run with base config only"

# Launch jobs for each delta config
for config in "${DELTA_CONFIGS[@]}"; do
    job_name="cosmx1k-${config%.yaml}"
    config_args="--config base_config.yaml --config delta_config/$config"
    description="Delta config variant: $config"
    
    submit_job "$job_name" "$config_args" "$description"
done

echo "All jobs submitted!"
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: $EXPERIMENT_DIR/slurm_logs/"
echo
echo "Expected wandb run names:"
echo "  - cosmx1k-baseline-corealigned (baseline with core-aligned data)"
for config in "${DELTA_CONFIGS[@]}"; do
    case "$config" in
        "disable_cell_level_config.yaml") echo "  - cosmx1k-no-cell-level" ;;
        "finetune_geneformer_config.yaml") echo "  - cosmx1k-finetune-uul" ;;
        "finetune_uni_config.yaml") echo "  - cosmx1k-finetune-luu" ;;
        "good_quality_cells_config.yaml") echo "  - cosmx1k-good-quality-only" ;;
        "nonaligned_cores_config.yaml") echo "  - cosmx1k-nonaligned-cores" ;;
        "ramp_up_cnn_config.yaml") echo "  - cosmx1k-ramped-cnn" ;;
        "cell_cnn_only_config.yaml") echo "  - cosmx1k-cell-cnn-only" ;;
    esac
done
