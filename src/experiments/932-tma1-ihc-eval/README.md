# TMA Gene Expression Decoder - IHC Validation

This experiment trains a gene expression decoder on frozen CellWhisperer embeddings to predict gene expression from histology images of TMA (Tissue Microarray) samples.

## Overview

**Goal**: Train a decoder to predict gene expression from images and validate predictions against IHC (immunohistochemistry) quantification.

**Approach**:
1. Train decoder on TMA1, TMA4, TMA5, TMA11_12, TMA13_14, TMA15_16
2. Evaluate on TMA2 (validation) and TMA3 (test)
3. Predict expression for TMA1, TMA2, TMA3 for downstream IHC validation

## Pipeline Structure

```
├── Snakefile                    # Main pipeline orchestration
├── scripts/
│   ├── train_decoder.py         # Train gene expression decoder
│   ├── predict_expression.py    # Predict gene expression for a TMA
│   └── evaluate_metrics.py      # Compute evaluation metrics
├── results/
│   ├── models/
│   │   └── decoder.ckpt        # Trained decoder checkpoint
│   ├── predictions/
│   │   ├── TMA1_predictions.h5ad
│   │   ├── TMA2_predictions.h5ad
│   │   └── TMA3_predictions.h5ad
│   ├── metrics/
│   │   ├── TMA2_metrics.csv    # Validation metrics
│   │   └── TMA3_metrics.csv    # Test metrics
│   └── plots/
│       ├── TMA2/               # Validation plots
│       └── TMA3/               # Test plots
└── patient_ihc.xlsx            # IHC quantification data
```

## Configuration

### Model
- **CellWhisperer checkpoint**: `spatialwhisperer_v1.ckpt`
- **Decoder architecture**: Single linear layer (1024 → ~6k genes)
- **Loss function**: MSE (configurable: mse/mae/huber)

### Training Datasets
- TMA1, TMA4, TMA5, TMA11_12, TMA13_14, TMA15_16

### Evaluation Datasets
- **Validation**: TMA2
- **Test**: TMA3

### Hyperparameters
- Batch size: 64
- Max epochs: 50
- Learning rate: 1e-3
- Optimizer: AdamW with cosine annealing

## Usage

### Prerequisites

Ensure the TMA datasets are prepared:
```bash
cd src/datasets/lymphoma_cosmx_large
pixi run --no-progress snakemake --cores 8
```

This creates the individual TMA h5ad files in `results/lymphoma_cosmx_large/h5ads/`.

### Running the Pipeline

```bash
cd src/experiments/932-tma1-ihc-eval

# Dry run to check everything
snakemake -n

# Run full pipeline (uses conda environment)
snakemake --cores 10 --use-conda

# Run specific rules
snakemake --cores 10 --use-conda results/models/decoder.ckpt  # Train only
snakemake --cores 10 --use-conda results/predictions/TMA1_predictions.h5ad  # Predict TMA1
snakemake --cores 10 --use-conda results/metrics/TMA2_metrics.csv  # Evaluate TMA2
```

### On Sherlock Cluster

The Snakefile includes SLURM resource specifications. Submit jobs with:

```bash
# Train decoder (requires GPU)
snakemake --cores 1 --use-conda results/models/decoder.ckpt \
    --cluster "sbatch -p gpu --gres=gpu:1 -c {threads} --mem={resources.mem_mb}MB" \
    --jobs 1

# Run all predictions and evaluations
snakemake --cores 5 --use-conda \
    --cluster "sbatch -p gpu --gres=gpu:1 -c {threads} --mem={resources.mem_mb}MB" \
    --jobs 5
```

## Outputs

### 1. Trained Model
`results/models/decoder.ckpt` - Trained gene expression decoder

### 2. Predictions
For each TMA (TMA1, TMA2, TMA3):
- `results/predictions/{TMA}_predictions.h5ad` - AnnData with:
  - `.X`: Predicted gene expression (log-transformed)
  - `.layers['ground_truth']`: True gene expression for comparison
  - `.obs`: Cell metadata from original dataset
  - `.obsm['spatial']`: Spatial coordinates (if available)
  - `.var`: Gene names

### 3. Evaluation Metrics
For TMA2 (validation) and TMA3 (test):
- `results/metrics/{TMA}_metrics.csv` - Summary metrics:
  - MSE, RMSE, MAE
  - Mean/median/std per-sample Pearson correlation
  - Mean/median/std per-gene Pearson correlation

### 4. Evaluation Plots
For each evaluated TMA in `results/plots/{TMA}/`:
- `scatter_pred_vs_gt.png` - Predicted vs ground truth scatter
- `hist_sample_correlations.png` - Distribution of per-sample correlations
- `hist_gene_correlations.png` - Distribution of per-gene correlations
- `top_bottom_genes.png` - Top and bottom 20 genes by correlation
- `gene_correlations.csv` - Per-gene correlation values

## Metrics Explained

### Overall Metrics
- **MSE**: Mean squared error between predicted and true expression
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error

### Per-Sample Metrics
- Pearson correlation computed for each cell (sample) across all genes
- Reports mean, median, and std of correlations across all samples
- Indicates how well the model captures per-cell expression patterns

### Per-Gene Metrics
- Pearson correlation computed for each gene across all cells
- Reports mean, median, and std of correlations across all genes
- Indicates which genes are well-predicted vs poorly-predicted

## Next Steps: IHC Validation

The predictions can be validated against IHC quantification in `patient_ihc.xlsx`:

1. Extract predictions for specific proteins quantified in IHC
2. Aggregate predictions at the patient/core level
3. Compute correlation between predicted expression and IHC quantification
4. Identify discordant cases for further analysis

(Implementation pending)

## Troubleshooting

### Missing CellWhisperer Checkpoint
If `spatialwhisperer_v1.ckpt` doesn't exist:
```bash
# Check available models
ls results/models/jointemb/

# Update MODEL_NAME in Snakefile if using a different checkpoint
```

### Dataset Not Found
Ensure TMA datasets are processed:
```bash
ls results/lymphoma_cosmx_large/h5ads/
# Should see: full_data_TMA1.h5ad, full_data_TMA2.h5ad, etc.
```

### Out of Memory
Reduce batch size in Snakefile:
```python
BATCH_SIZE = 32  # or 16
```

### SLURM Resource Issues
Adjust resources in the Snakefile rules:
```python
resources:
    mem_mb = 80000,  # Reduce if needed
    slurm = slurm_gres("medium", num_cpus=10)  # Use medium instead of large
```

## File Descriptions

### Snakefile
Main workflow orchestration defining:
- Training rule: Trains decoder on specified TMAs
- Prediction rules: Generates predictions for each TMA
- Evaluation rules: Computes metrics comparing predictions to ground truth
- Summary rule: Creates summary report

### scripts/train_decoder.py
Trains the gene expression decoder:
- Loads frozen CellWhisperer model
- Initializes decoder (single linear layer by default)
- Trains on specified TMA datasets
- Saves best model checkpoint

### scripts/predict_expression.py
Runs inference on a TMA dataset:
- Loads trained decoder and CellWhisperer
- Extracts image embeddings
- Predicts gene expression
- Saves predictions with ground truth for comparison

### scripts/evaluate_metrics.py
Computes evaluation metrics:
- Loads predictions and ground truth
- Computes MSE, MAE, correlations
- Creates visualization plots
- Saves metrics to CSV

## References

- CellWhisperer: Multi-modal foundation model for spatial transcriptomics
- Gene expression decoder: Linear probe on frozen embeddings
- TMA datasets: Lymphoma CosMx large dataset (12 TMAs)
