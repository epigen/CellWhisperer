# Gene Expression Decoder

This module trains a decoder on top of frozen CellWhisperer embeddings to predict gene expression from images.

## Overview

The decoder takes image embeddings from a trained CellWhisperer model and predicts log-transformed gene expression for all ~6k genes in the dataset.

**Architecture:**
- **Input:** Image embeddings from CellWhisperer (1024-dim by default)
- **Decoder:** Single linear layer (by default) mapping embeddings → gene expression
- **Output:** Predicted log(expression+1) for all genes

## Files

- `gene_expression_decoder.py`: Decoder model definition
- `gene_expression_decoder_lightning.py`: Lightning training wrapper
- `train_decoder.py`: Training script using Lightning CLI

## Usage

### Basic Training

```bash
pixi run --no-progress python -m cellwhisperer.expression_decoder.train_decoder \
    --model.cellwhisperer_checkpoint path/to/cellwhisperer.ckpt \
    --data.dataset_names quilt1m \
    --data.batch_size 64 \
    --trainer.max_epochs 50
```

### Advanced Configuration

```bash
pixi run --no-progress python -m cellwhisperer.expression_decoder.train_decoder \
    --model.cellwhisperer_checkpoint path/to/checkpoint.ckpt \
    --model.decoder_config.gene_list_path path/to/gene_list.csv \
    --model.decoder_config.embedding_dim 1024 \
    --model.decoder_config.hidden_dims [] \
    --model.learning_rate 1e-3 \
    --model.loss_type mse \
    --data.dataset_names quilt1m,hest1k \
    --data.batch_size 64 \
    --data.nproc 8 \
    --trainer.max_epochs 50 \
    --trainer.accelerator gpu \
    --trainer.devices 1
```

### Key Parameters

**Model parameters:**
- `cellwhisperer_checkpoint`: Path to trained CellWhisperer checkpoint
- `decoder_config.gene_list_path`: Path to gene list CSV (default: uses cosmx6k_genes from config)
- `decoder_config.embedding_dim`: Embedding dimension (default: 1024)
- `decoder_config.hidden_dims`: List of hidden layer sizes (default: [] for single layer)
- `learning_rate`: Learning rate (default: 1e-3)
- `loss_type`: Loss function - "mse", "mae", or "huber" (default: "mse")

**Data parameters:**
- `dataset_names`: Comma-separated dataset names
- `batch_size`: Batch size for training
- `nproc`: Number of dataloader workers

## Implementation Details

### Frozen CellWhisperer
The CellWhisperer model is loaded from checkpoint and all parameters are frozen. Only the decoder head is trained.

### Loss Functions
- **MSE** (mean squared error): Default, good for regression
- **MAE** (mean absolute error): More robust to outliers
- **Huber**: Combines MSE and MAE benefits

### Metrics
The module logs two metrics during training:
1. **Loss**: The selected loss function value
2. **Correlation**: Per-sample Pearson correlation between predicted and true expression

### Gene List
The number of genes is automatically determined by reading the gene list CSV file:
- Default: Uses `cosmx6k_genes` path from config (typically ~6k genes)
- Custom: Specify `decoder_config.gene_list_path` to use a different gene set
- The CSV must have a `gene_name` column

### Data Requirements
The dataloader must provide:
- **Image data** (patches_ctx, patches_cell) for embedding extraction
- **Gene expression** (expression_expr) containing log(counts+1) for all genes in the gene list

## Example Training Flow

1. **Load frozen CellWhisperer**: Checkpoint is loaded and all parameters frozen
2. **Extract embeddings**: For each batch, image embeddings are computed (no gradients)
3. **Predict expression**: Decoder predicts gene expression from embeddings
4. **Compute loss**: Compare predicted vs ground truth log-transformed expression
5. **Update decoder**: Only decoder parameters are updated via backprop

## Extending the Architecture

To use a deeper decoder with hidden layers:

```bash
pixi run --no-progress python -m cellwhisperer.expression_decoder.train_decoder \
    --model.decoder_config.hidden_dims [2048,4096] \
    --model.decoder_config.dropout_rate 0.1 \
    --model.decoder_config.activation relu \
    ...
```

This creates: embedding_dim → 2048 → 4096 → num_genes with ReLU and dropout.
