#!/bin/bash
#SBATCH --job-name train_jointemb
#SBATCH --partition=gpu
#SBATCH --qos=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --time=03-00:00:00

node=$(hostname --long)
user=$(whoami)

source /msc/home/mschae83/.bashrc
# conda info --envs
source activate single-cellm-moritz2

cd /msc/home/mschae83/single-cellm/src/single_cellm/jointemb
python training.py
