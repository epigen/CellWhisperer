#!/bin/bash

# TODO maybe this script is unnecessary

source /msc/home/mschae83/miniconda3/etc/profile.d/conda.sh
source activate textgen

export GLOBAL_RANK=$SLURM_LOCALID
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_LOCALID

# Move into the correct directory, especially for wandb to pick up the git commit
cd $HOME/text-generation-webui
# start training (or whatever)

./start_linux.sh
