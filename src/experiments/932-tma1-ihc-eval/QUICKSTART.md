# Quick Start Guide

## TL;DR - Run the Full Pipeline

```bash
cd /home/moritz/code/cellwhisperer/src/experiments/932-tma1-ihc-eval

# Check what will run
snakemake -n

# Run everything (uses cellwhisperer conda environment)
snakemake --cores 10 --use-conda
```

## Step-by-Step

### 1. Verify Prerequisites

Check that TMA datasets are prepared:
```bash
ls /home/moritz/code/cellwhisperer/results/lymphoma_cosmx_large/h5ads/
```

Should see: `full_data_TMA1.h5ad`, `full_data_TMA2.h5ad`, etc.

If not, prepare them:
```bash
cd /home/moritz/code/cellwhisperer/src/datasets/lymphoma_cosmx_large
pixi run --no-progress snakemake --cores 8
```

### 2. Check CellWhisperer Model

Verify the model checkpoint exists:
```bash
ls /home/moritz/code/cellwhisperer/results/models/jointemb/spatialwhisperer_v1.ckpt
```

If using a different model, update `MODEL_NAME` in the Snakefile.

### 3. Run Pipeline

```bash
cd /home/moritz/code/cellwhisperer/src/experiments/932-tma1-ihc-eval

# Dry run (see what will execute)
snakemake -n

# Run full pipeline (uses cellwhisperer conda environment)
snakemake --cores 10 --use-conda
```

### 4. Check Results

After completion:
```bash
# Trained model
ls results/models/decoder.ckpt

# Predictions
ls results/predictions/

# Metrics
cat results/metrics/TMA2_metrics.csv
cat results/metrics/TMA3_metrics.csv

# Plots
ls results/plots/TMA2/
ls results/plots/TMA3/
```

## Running Individual Steps

### Train Decoder Only
```bash
snakemake --cores 10 --use-conda results/models/decoder.ckpt
```

### Predict for Specific TMA
```bash
snakemake --cores 10 --use-conda results/predictions/TMA1_predictions.h5ad
```

### Evaluate Specific TMA
```bash
snakemake --cores 10 --use-conda results/metrics/TMA2_metrics.csv
```

## Expected Runtime

On a single GPU:
- **Training**: ~2-4 hours (50 epochs)
- **Prediction per TMA**: ~10-30 minutes
- **Evaluation per TMA**: ~5 minutes

**Total**: ~3-5 hours for full pipeline

## Common Issues

### "No such file or directory" for TMA datasets
Run the dataset preparation step (see Step 1 above)

### "Checkpoint not found"
Update `MODEL_NAME` in Snakefile to match available model in `results/models/jointemb/`

### Out of memory
Edit Snakefile and reduce `BATCH_SIZE` from 64 to 32 or 16

## Outputs Summary

- **Model**: `results/models/decoder.ckpt`
- **Predictions**: `results/predictions/{TMA}_predictions.h5ad` (TMA1, TMA2, TMA3)
- **Metrics**: `results/metrics/{TMA}_metrics.csv` (TMA2, TMA3)
- **Plots**: `results/plots/{TMA}/` (scatter, histograms, top genes)

See README.md for detailed documentation.
