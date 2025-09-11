# SpotWhisperer Evaluation Pipeline

This directory contains a comprehensive Snakemake pipeline for training and evaluating SpotWhisperer models on different dataset combinations.

## Overview

The pipeline systematically trains SpotWhisperer models on various combinations of paired datasets and evaluates each trained model using three different benchmarks.

## Dataset Combinations

The pipeline trains models on the following dataset combinations:

### Individual datasets:
- `cellxgene_census+archs4_geo` 
- `hest1k`
- `quilt1m`

### Pairs of datasets:
- `cellxgene_census+archs4_geo` + `hest1k`
- `cellxgene_census+archs4_geo` + `quilt1m` 
- `hest1k` + `quilt1m`

### All three datasets:
- `cellxgene_census+archs4_geo` + `hest1k` + `quilt1m`

Each combination is trained with 3 different seeds (0, 1, 2) for reproducibility.

## Evaluation Benchmarks

Each trained model is evaluated using three benchmarks:

1. **HEST Benchmark** (`rules/hest_benchmark.smk`)
   - Spatial transcriptomics prediction tasks
   - Uses existing HEST evaluation framework

2. **Lung Tissue Benchmark** (`rules/lung_benchmark.smk`) 
   - Lung tissue-specific evaluation
   - Tests on lung tissue datasets

3. **CellWhisperer Benchmark** (`rules/cellwhisperer_benchmark.smk`)
   - Zero-shot cell type, disease, and cell origin predictions
   - Reimplements validation from `src/figures/fig2_embedding_validations.smk`
   - Uses metadata columns defined in `metadata_cols_per_zero_shot_validation_dataset` from config.yaml
   - Tests on: tabula_sapiens, pancreas, aida, human_disease, immgen datasets

## Usage

### Run full experiment:
```bash
cd src/spotwhisperer_eval
snakemake --cores 16 -j 8 all
```

### Run with fast development mode:
```bash
cd src/spotwhisperer_eval
snakemake --cores 16 -j 8 all --config fast=true
```

### Dry run to check pipeline structure:
```bash
snakemake --dry-run -j1 all
```

## Configuration

The pipeline uses:
- **Base config**: `src/spotwhisperer_v2.yaml`
- **Dataset override**: `--data.dataset_names` parameter is dynamically set for each combination
- **Seeds**: Currently using seed 0 for efficiency
- **Fast training**: Set `fast=true` in config to enable `--trainer.fast_dev_run` for quick development
- **Resources**: Configured for large GPU nodes with substantial memory requirements

## Output Structure

Results are organized as follows:

```
results/spotwhisperer_eval/
├── models/                      # Trained models
│   ├── {dataset_combo}/
│   │   └── seed_{seed}.ckpt
├── benchmarks/                  # Evaluation results
│   ├── hest/{dataset_combo}/seed_{seed}/
│   ├── lung/{dataset_combo}/seed_{seed}/
│   ├── cellwhisperer/{dataset_combo}/seed_{seed}/
│   ├── aggregated_results.csv   # Combined results
│   └── performance_summary.png  # Summary visualization
```

## Key Features

- **Systematic dataset combinations**: Tests all possible 1, 2, and 3-dataset combinations
- **Multiple evaluation methods**: Three different benchmarks provide comprehensive assessment
- **Reproducibility**: Multiple seeds and deterministic pipeline execution
- **Canonical Snakemake patterns**: Proper input/output dependencies, resource management
- **Notebook-based analysis**: Uses Jupyter notebooks for complex analysis tasks
- **Comprehensive results aggregation**: Combines all benchmark results for comparison

## Development Notes

For accelerated development runs:
- Use `fast=true` config option to enable fast development mode with `--trainer.fast_dev_run`
- Pipeline currently uses single seed (0) for efficiency
- The pipeline is designed to be scalable - start small and expand as needed

## Analysis Questions

The pipeline is designed to answer:
- How does retrieval and benchmark performance vary with different dataset combinations?
- What is the performance of held-out dataset pairs compared to:
  - Random baselines (trained on one modality pair)
  - Full-trained models  
  - Matching modality pair models?
- How consistent are results across different random seeds?
