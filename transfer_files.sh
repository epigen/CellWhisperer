#!/bin/bash

# Create a script to transfer files from muwhpc to lustre
# This script will copy the transfered_labels.csv file for the aida dataset

# Ensure the target directory exists on lustre
ssh lustre "mkdir -p /nobackup/lab_bock/public_html/papers/schaefer2025cellwhisperer/datasets/aida/finetuning_eval/"

# Copy the file from muwhpc to lustre
scp muwhpc:/msc/home/q56ppene/cellwhisperer_revision2/cellwhisperer_private/results/finetuning_eval/aida/transfered_labels.csv \
    lustre:/nobackup/lab_bock/public_html/papers/schaefer2025cellwhisperer/datasets/aida/finetuning_eval/transfered_labels.csv

echo "File transfer completed"
