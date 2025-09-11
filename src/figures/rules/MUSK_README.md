# MUSK Evaluation Pipeline

This document describes the new MUSK evaluation pipeline implemented in `src/figures/rules/spotwhisperer-musk.smk`.

## Overview

The MUSK evaluation pipeline is designed to run comprehensive evaluations on pathology image benchmarks using the MUSK framework. It includes the following evaluation tasks:

- **Zero-shot image-text retrieval**: Evaluate cross-modal retrieval capabilities
- **Zero-shot classification**: Test classification performance without training
- **Image-to-image retrieval**: Assess similarity-based image retrieval
- **Few-shot classification**: Evaluate performance with limited training examples
- **Linear probe classification**: Full dataset supervised classification

## Pipeline Structure

The pipeline follows the same structure as `spotwhisperer.smk` and includes these main rules:

### Core Evaluation Rules

1. `setup_musk_models`: Prepares model configuration file
2. `musk_zero_shot_retrieval`: Cross-modal retrieval evaluation
3. `musk_zero_shot_classification`: Zero-shot classification evaluation
4. `musk_image_retrieval`: Image-to-image retrieval evaluation
5. `musk_few_shot_classification`: Few-shot learning evaluation
6. `musk_linear_probe`: Full dataset linear probe evaluation
7. `musk_performance_summary`: Aggregates and summarizes all results
8. `musk_all`: Main rule that runs the complete pipeline

## Configuration

### Dataset Root

The dataset location can be configured by setting `musk_dataset_root` in your `config.yaml`:

```yaml
musk_dataset_root: "/path/to/downstreams_demo"
```

If not specified, it defaults to `/tmp/downstreams_demo`.

### Model Configuration

Currently, the pipeline uses the MUSK model (`musk_large_patch16_384` from `hf_hub:xiangjx/musk`). To use a different model, modify the `setup_musk_models` rule parameters.


## Results

Results are organized in the following structure:

```
results/plots/musk_evaluation/{model}/
├── models.txt                          # Model configuration
├── results/                            # Individual evaluation results (JSON)
│   ├── zeroshot_retrieval_*.json
│   ├── zeroshot_classification_*.json
│   ├── image_retrieval_*.json
│   ├── fewshot_classification_*.json
│   └── linear_probe_*.json
└── summary/                            # Aggregated summaries
    └── musk_evaluation_summary.json
```

## Datasets Evaluated

The pipeline evaluates on the following datasets:

- **pathmmu_retrieval**: Multi-modal pathology retrieval
- **unitopatho_retrieval**: UniToPatho image retrieval
- **skin**: Skin pathology classification
- **pannuke**: PanNuke nuclei classification  
- **unitopatho**: UniToPatho classification

## Dependencies

The pipeline requires:

- The `cellwhisperer` conda environment
- MUSK benchmarks code in `src/experiments/MUSK/benchmarks/`
- Access to the evaluation datasets

## Performance Summary

The pipeline generates a comprehensive performance summary including:

- Task-specific performance metrics
- Visualizations of results across datasets
- Structured JSON output for further analysis

## Adaptation for SpotWhisperer

**Note**: The current implementation uses the original MUSK model since it's designed for pathology image tasks. To use SpotWhisperer (which is designed for spatial transcriptomics), you would need to:

1. Adapt SpotWhisperer to work with pathology images, or
2. Create a bridge/adapter layer to make it compatible with the MUSK evaluation framework, or
3. Modify the evaluation framework to work with transcriptomics data

This adaptation is beyond the scope of the current implementation but the pipeline structure supports easy model swapping through the `setup_musk_models` rule.

## Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure `musk_dataset_root` points to the correct location
2. **GPU memory issues**: Reduce batch sizes in rule parameters
3. **Missing dependencies**: Ensure the `cellwhisperer` environment includes all required packages

### Logs

Individual evaluation logs are stored in `logs/musk_*.log` files for debugging purposes.