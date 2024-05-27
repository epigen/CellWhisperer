#!/bin/bash
#SBATCH -p gpu # partition (queue)
#SBATCH -q a100
#SBATCH --gres=gpu:a100:1
#SBATCH -c 1 # number of cores
#SBATCH --mem 300G # memory 
#SBATCH -t 3-0:00 # time (D-HH:MM)
#SBATCH -o slurm.celllwhisperer_%N.%j.out # STDOUT
#SBATCH -e slurm.celllwhisperer_%N.%j.err # STDERR


## SBATCH -p gpu # partition (queue)
## SBATCH -q a100-sxm4-80gb
## SBATCH --gres=gpu:a100-sxm4-80gb:1
## SBATCH -c 1 # number of cores
## SBATCH --mem 450G # memory 
## SBATCH -t 3-0:00 # time (D-HH:MM)
## SBATCH -o slurm.snakemake_cfdna_%N.%j.out # STDOUT
## SBATCH -e slurm.snakemake_cfdna_%N.%j.err # STDERR

# Load the conda environment
source activate  /msc/home/q56ppene/miniconda3/envs/cellwhisperer3/

# Run the validation script
python /msc/home/q56ppene/cellwhisperer/cellwhisperer/src/validation/zero_shot/MS_zero_shot_scfoundation_replication/run_validations.py

# Deactivate the conda environment
conda deactivate
