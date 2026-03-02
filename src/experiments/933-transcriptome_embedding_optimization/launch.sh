#!/bin/bash

# Transcriptome Embedding Optimization Experiment - SLURM Launch Script
# Compares multiple transcriptome models: UCE4/UCE33 (frozen/fine-tuned), Geneformer (frozen/fine-tuned), MLP

# Dynamically determine project root and experiment directory
PROJECT_ROOT=$(git rev-parse --show-toplevel)
EXPERIMENT_DIR="$PROJECT_ROOT/src/experiments/933-transcriptome_embedding_optimization"

# Detect conda installation path via multiple methods
CONDA_PROFILE=""
CONDA_ROOT=""
CONDA_ENV=""

# Method 1: Use environment variables from server (if available)
if [ -n "$_CONDA_ROOT" ]; then
    CONDA_ROOT="$_CONDA_ROOT"
    CONDA_PROFILE="$_CONDA_ROOT/etc/profile.d/conda.sh"
elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
    CONDA_ROOT="$CONDA_PREFIX"
    CONDA_PROFILE="$CONDA_PREFIX/etc/profile.d/conda.sh"
elif [ -n "$CONDA_EXE" ]; then
    CONDA_ROOT="$(dirname "$(dirname "$CONDA_EXE")")"
    CONDA_PROFILE="$CONDA_ROOT/etc/profile.d/conda.sh"
fi

# Method 2: Fallback to common installation paths
if [ -z "$CONDA_PROFILE" ] || [ ! -f "$CONDA_PROFILE" ]; then
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
        CONDA_ROOT="$(dirname "$(dirname "$conda_path")")"
        CONDA_PROFILE="$CONDA_ROOT/etc/profile.d/conda.sh"
    fi
fi

# Verify conda.sh exists
if [ -z "$CONDA_PROFILE" ] || [ ! -f "$CONDA_PROFILE" ]; then
    echo "Error: Could not detect conda installation"
    exit 1
fi

# Detect appropriate conda environment
if [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE" 2>/dev/null
    if command -v conda >/dev/null 2>&1; then
        if conda env list 2>/dev/null | grep -q "cellwhisperer-aarch64"; then
            CONDA_ENV="cellwhisperer-aarch64"
        elif conda env list 2>/dev/null | grep -q "cellwhisperer"; then
            CONDA_ENV="cellwhisperer"
        else
            CONDA_ENV="cellwhisperer"
        fi
    else
        CONDA_ENV="cellwhisperer"
    fi
else
    CONDA_ENV="cellwhisperer"
fi

# Parse optional experiment label argument
EXPERIMENT_LABEL="${1:-}"

# Array of delta config files
declare -a DELTA_CONFIGS=(
    "uce4_frozen.yaml"
    "uce4_finetune.yaml"
    "uce33_frozen.yaml"
    "geneformer_frozen.yaml"
    "geneformer_finetune.yaml"
    "census_cosmx_tma5_filtered.yaml"
    "census_cosmx_tma5_unfiltered.yaml"
)

# Validate label if provided
if [ -n "$EXPERIMENT_LABEL" ]; then
    valid=false
    if [ "$EXPERIMENT_LABEL" = "base_config" ]; then
        valid=true
    else
        for config in "${DELTA_CONFIGS[@]}"; do
            if [ "${config%.yaml}" = "$EXPERIMENT_LABEL" ]; then
                valid=true
                break
            fi
        done
    fi
    if [ "$valid" = false ]; then
        echo "Error: Unknown experiment label '$EXPERIMENT_LABEL'"
        echo "Valid labels: base_config ${DELTA_CONFIGS[*]%.yaml}"
        exit 1
    fi
    echo "Launching single experiment: $EXPERIMENT_LABEL"
else
    echo "Launching all experiments"
fi

echo "Project root: $PROJECT_ROOT"
echo "Experiment directory: $EXPERIMENT_DIR"
echo "Conda environment: $CONDA_ENV"
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
    
    sbatch_script="$EXPERIMENT_DIR/slurm_logs/sbatch_${job_name}.job"
    
    cat > "$sbatch_script" << EOF
#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --time=48:00:00
#SBATCH --partition=cmackall
#SBATCH --mem=200000
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
    
    jobid=$(sbatch "$sbatch_script" | awk '{print $4}')
    
    echo "  Job submitted with ID: $jobid"
    echo "  Log file: slurm_logs/slurm-${jobid}-${job_name}.out"
    echo
    
    sleep 2
}

# Launch baseline job (base config only - MLP with cosmx6k filtering)
if [ -z "$EXPERIMENT_LABEL" ] || [ "$EXPERIMENT_LABEL" = "base_config" ]; then
    submit_job "emb-opt-baseline" "--config base_config.yaml" "Baseline MLP run"
fi

# Launch jobs for each delta config
for config in "${DELTA_CONFIGS[@]}"; do
    if [ -n "$EXPERIMENT_LABEL" ] && [ "${config%.yaml}" != "$EXPERIMENT_LABEL" ]; then
        continue
    fi
    job_name="emb-opt-${config%.yaml}"
    config_args="--config base_config.yaml --config delta_config/$config"
    description="Delta config variant: $config"
    
    submit_job "$job_name" "$config_args" "$description"
done

echo "All jobs submitted!"
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: $EXPERIMENT_DIR/slurm_logs/"
echo
echo "Expected wandb run names:"
echo "  - transcriptome-emb-opt-mlp (baseline MLP)"
echo "  - transcriptome-emb-opt-uce4-frozen"
echo "  - transcriptome-emb-opt-uce4-finetune"
echo "  - transcriptome-emb-opt-uce33-frozen"
echo "  - transcriptome-emb-opt-geneformer-frozen"
echo "  - transcriptome-emb-opt-geneformer-finetune"
