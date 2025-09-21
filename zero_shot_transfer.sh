#!/bin/bash

# Script to transfer zero-shot LLM prediction files from muwhpc to lustre
# Simplified version for specific models and datasets

# Define specific datasets and models
datasets=("aida" "pancreas" "tabula_sapiens_well_studied_celltypes" "human_disease" "tabula_sapiens" "immgen" "tabula_sapiens_100_cells_per_type")
models=("claudesonnet" "llama33" "mistral7b" "gpt4")
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
                # Source and target paths
                source_file="${source_base}/${model}/datasets/${dataset}/predictions/${metadata_col}.${grouping}.csv"
                target_file="${target_base}/${dataset}/zero_shot_llm/${model}_${metadata_col}_${grouping}.csv"
                
                # Copy the file from muwhpc to lustre (suppress errors)
                scp -q "muwhpc:${source_file}" "lustre:${target_file}" 2>/dev/null || true
            done
        done
    done
done

echo "File transfers completed"
