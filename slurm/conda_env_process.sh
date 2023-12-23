#!/bin/bash

# export OMP_NUM_THREADS=1  # TODO should I ?

source /msc/home/mschae83/miniconda3/etc/profile.d/conda.sh
source activate single-cellm

export GLOBAL_RANK=$SLURM_LOCALID
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_LOCALID

# Move into the correct directory, especially for wandb to pick up the git commit
cd $HOME/single-cellm
# start training (or whatever)
"$@"
