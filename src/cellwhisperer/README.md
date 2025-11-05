# CellWhisperer Python Package

This directory contains the core Python package for the CellWhisperer embedding model.

## Structure

- **`jointemb/`**: Joint embedding model implementation
  - Model architecture combining transcriptome and text encoders
  - Training code using PyTorch Lightning
  - Inference utilities
  - Configuration and hyperparameter management

- **`validation/`**: Validation and evaluation utilities
  - Metrics for assessing model performance
  - Benchmark scripts
  - Analysis tools for model outputs

- **`utils/`**: Shared utility functions
  - Data processing helpers
  - Common transformations
  - Helper functions used across modules

- **`misc/`**: Miscellaneous utilities and scripts

- **`config.py`**: Global configuration management

## Usage

The main entry point for training is accessible via the `cellwhisperer` command after installation:

```bash
conda activate cellwhisperer
cellwhisperer fit --config <config_file.yaml>
```

For more details on training, see the main repository README and the configuration examples in `src/cellwhisperer_clip_v1.yaml`.

## Model Architecture

CellWhisperer uses a contrastive learning approach to align transcriptome representations with natural language descriptions. The model consists of:

1. **Transcriptome Encoder**: Based on Geneformer, processes gene expression data
2. **Text Encoder**: Based on BioBERT, processes natural language annotations
3. **Joint Embedding Space**: Learned shared representation space enabling cross-modal retrieval

The model is trained to maximize similarity between matched transcriptome-text pairs while minimizing similarity for unmatched pairs.
