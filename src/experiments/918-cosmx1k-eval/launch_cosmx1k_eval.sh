#!/bin/bash

# CosMx1K Evaluation Experiment (918) - SLURM Launch Script
# This script launches all experimental variants as separate SLURM jobs

# Dynamically determine project root and experiment directory
PROJECT_ROOT=$(git rev-parse --show-toplevel)
EXPERIMENT_DIR="$PROJECT_ROOT/src/experiments/918-cosmx1k-eval"

# Detect conda installation path dynamically
if [ -f "$HOME/.mamba/etc/profile.d/conda.sh" ]; then
    CONDA_PROFILE="$HOME/.mamba/etc/profile.d/conda.sh"
    CONDA_ENV="cellwhisperer-aarch64"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    CONDA_PROFILE="$HOME/miniforge3/etc/profile.d/conda.sh"
    CONDA_ENV="cellwhisperer"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PROFILE="$HOME/miniconda3/etc/profile.d/conda.sh"
    CONDA_ENV="cellwhisperer"
else
    echo "Error: Could not find conda installation"
    exit 1
fi

# Array of delta config files
declare -a DELTA_CONFIGS=(
    "disable_cell_level_config.yaml"
    "finetune_geneformer_config.yaml" 
    "finetune_geneformer_luu_config.yaml"
    "good_quality_cells_config.yaml"
    "improved_alignment_config.yaml"
    "ramp_up_cnn_config.yaml"
)

echo "Launching CosMx1K evaluation jobs..."
echo "Project root: $PROJECT_ROOT"
echo "Experiment directory: $EXPERIMENT_DIR"
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
#SBATCH --mem=50000
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
echo "  - cosmx1k-baseline (baseline)"
for config in "${DELTA_CONFIGS[@]}"; do
    case "$config" in
        "disable_cell_level_config.yaml") echo "  - cosmx1k-no-cell-level" ;;
        "finetune_geneformer_config.yaml") echo "  - cosmx1k-finetune-uul" ;;
        "finetune_geneformer_luu_config.yaml") echo "  - cosmx1k-finetune-luu" ;;
        "good_quality_cells_config.yaml") echo "  - cosmx1k-good-quality-only" ;;
        "improved_alignment_config.yaml") echo "  - cosmx1k-core-aligned" ;;
        "ramp_up_cnn_config.yaml") echo "  - cosmx1k-ramped-cnn" ;;
    esac
done