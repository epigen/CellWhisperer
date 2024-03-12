#!/bin/bash
#
#SBATCH -p cpu # partition (queue)
#SBATCH -c 3 # number of cores
#SBATCH --mem 1400G # memory 
#SBATCH -t 3-0:00 # time (D-HH:MM)
#SBATCH -o slurm.snakemake_cellxgene_%N.%j.out # STDOUT
#SBATCH -e slurm.snakemake_cellxgene_%N.%j.err # STDERR

cd /msc/home/q56ppene/cellwhisperer/cellwhisperer
source activate cellwhisperer
snakemake --snakefile src/datasets/cellxgene_census/Snakefile -c 1 --keep-going --unlock
snakemake --snakefile src/datasets/cellxgene_census/Snakefile -c 1 --keep-going --rerun-triggers mtime