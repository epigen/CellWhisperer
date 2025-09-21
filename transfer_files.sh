#!/bin/bash

# Create a script to transfer files from muwhpc to lustre
# This script will copy the transfered_labels.csv files for multiple datasets

# List of datasets to process
datasets=("aida" "immgen" "pancreas" "tabula_sapiens" "tabula_sapiens_well_studied_celltypes")

# Process each dataset
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    
    # Ensure the target directory exists on lustre
    ssh lustre "mkdir -p /nobackup/lab_bock/public_html/papers/schaefer2025cellwhisperer/datasets/$dataset/finetuning_eval/"
    
    # Copy the file from muwhpc to lustre
    scp muwhpc:/msc/home/q56ppene/cellwhisperer_revision2/cellwhisperer_private/results/finetuning_eval/$dataset/transfered_labels.csv \
        lustre:/nobackup/lab_bock/public_html/papers/schaefer2025cellwhisperer/datasets/$dataset/finetuning_eval/transfered_labels.csv
    
    echo "Completed transfer for $dataset"
done

echo "All file transfers completed"
