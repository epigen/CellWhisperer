#!/bin/bash

# Script to transfer zero-shot LLM prediction files from muwhpc to lustre
# Based on the Snakemake rule zero_shot_llm_prediction_download

# Define datasets, models, metadata columns, and groupings
datasets=("aida" "immgen" "pancreas" "tabula_sapiens" "tabula_sapiens_well_studied_celltypes")
models=("gpt4" "llama33" "claudesonnet" "deepseek" "mistral7b")
metadata_cols=("cell_type" "cell_ontology_class")
groupings=("by_cell" "by_class")

# Base paths
source_base="/msc/home/q56ppene/cellwhisperer_revision2/cellwhisperer_private/results/plots/zero_shot_validations"
target_base="/nobackup/lab_bock/public_html/papers/schaefer2025cellwhisperer/datasets"

# Process each combination
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    
    # Create target directory on lustre
    ssh lustre "mkdir -p ${target_base}/${dataset}/zero_shot_llm/"
    
    for model in "${models[@]}"; do
        for metadata_col in "${metadata_cols[@]}"; do
            for grouping in "${groupings[@]}"; do
                echo "Transferring ${model}_${metadata_col}_${grouping}.csv for ${dataset}"
                
                # Source and target paths
                source_file="${source_base}/${model}/datasets/${dataset}/predictions/${metadata_col}.${grouping}.csv"
                target_file="${target_base}/${dataset}/zero_shot_llm/${model}_${metadata_col}_${grouping}.csv"
                
                # Copy the file from muwhpc to lustre
                scp "muwhpc:${source_file}" "lustre:${target_file}" || echo "Failed to transfer ${source_file}"
            done
        done
    done
    
    echo "Completed transfers for ${dataset}"
done

echo "All zero-shot LLM prediction file transfers completed"
