# Sherlock Deployment Guide - UCE Model Migration

This guide covers deploying the new UCE model implementation to Sherlock and running the UCE experiment.

## Overview

**Branch:** `uce-kuan`  
**Purpose:** Test new UCE implementation (KuanP/uce-cosmx-geneset) in lymphoma trimodal optimization experiment  
**Key Changes:**
- New UCE model using Kuan's implementation
- 512-dim embeddings (vs old 1280-dim)
- Simplified gene tokenization (no protein embeddings needed)
- Auto-downloads from HuggingFace

## Deployment Steps

### 1. Connect to Sherlock

```bash
ssh sherlock
```

### 2. Navigate to Project Directory

```bash
cd ~/cellwhisperer_private  # or wherever your clone is
```

### 3. Pull the Branch

```bash
git fetch origin
git checkout uce-kuan
git pull origin uce-kuan
```

### 4. Update Submodules

The new UCE implementation requires updating the `data_collection_exp` submodule:

```bash
git submodule update --init --recursive
```

Verify submodule is correct:
```bash
cd modules/data_collection_exp
git remote -v
# Should show:
#   origin: git@github.com:moritzschaefer/data_collection_exp.git
#   upstream: git@github.com:Kuan-Pang/data_collection_exp.git
cd ../..
```

### 5. Update Conda Environment

The new UCE model requires `transformers>=4.57.1` and the `data-collection-exp` package:

```bash
conda activate cellwhisperer

# Install updated dependencies
pip install 'transformers>=4.57.1'
pip install -e modules/data_collection_exp
```

**Note:** The conda environment on Sherlock should already have most dependencies. The key additions are:
- `transformers>=4.57.1` (for UCE model)
- `data-collection-exp` (Kuan's implementation)

### 6. Verify UCE Model Loads

Quick test to ensure the new UCE model can be imported and loaded:

```bash
conda run -n cellwhisperer python -c "
from cellwhisperer.jointemb.uce_model import UCEModel, UCEConfig
import torch

config = UCEConfig.from_pretrained('KuanP/uce-cosmx-geneset')
print(f'UCE config loaded. Output dim: {config.output_dim}')

model = UCEModel.from_pretrained('KuanP/uce-cosmx-geneset')
print(f'UCE model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

Expected output:
```
UCE config loaded. Output dim: 512
UCE model loaded. Total parameters: 773,xxx,xxx
```

### 7. Check Static Files

Verify gene data files are in place:

```bash
ls -lh static/UCE/
# Should show:
#   gene_names.txt (61,888 genes)
#   all_species_gene_dict.json (mapping data)
#   README.md (documentation)
```

### 8. Launch the UCE Experiment

Navigate to experiment directory and launch:

```bash
cd src/experiments/lymphoma_trimodal_optimization
bash launch_uce.sh  # Add --dry-run to test first
```

This will:
- Submit a SLURM job to partition `cmackall`
- Request 1 H100 GPU, 8 CPUs, 200GB RAM
- Run for up to 12 hours
- Log to `/scratch/users/moritzs/lymphoma_trimodal_opt_logs/uce_*.{out,err}`

### 9. Monitor the Job

```bash
# Check job status
squeue -u $USER

# View live log output
tail -f /scratch/users/moritzs/lymphoma_trimodal_opt_logs/uce_*.out

# Check for errors
tail -f /scratch/users/moritzs/lymphoma_trimodal_opt_logs/uce_*.err
```

### 10. Track Progress in Weights & Biases

The experiment logs to W&B:
- **Project:** `SpatialWhisperer`
- **Entity:** `single-cellm`
- **Group:** `lymphoma-trimodal-opt`
- **Run name:** `trimodal-uce`

Visit: https://wandb.ai/single-cellm/SpatialWhisperer

## Expected Results

The experiment will train for 4 epochs with:
- **Datasets:** lymphoma_cosmx_large (TMA2, TMA4, TMA5) + cellxgene_census
- **Training:** TMA4, TMA5, census (raw counts)
- **Validation:** TMA2 (raw counts)
- **Transcriptome encoder:** UCE (512-dim embeddings)
- **Text encoder:** BERT
- **Image encoder:** UNI2 + CNN (512-dim, 4 layers)
- **Precision:** bf16-mixed
- **Batch size:** 512

**Key metrics to watch:**
- `val_retrieval/transcriptome_image/rocauc_macroAvg` (checkpoint selection)
- `val_retrieval/image_transcriptome/rocauc_macroAvg`
- Zero-shot cell type annotation (TabSap, ImmGen, HumanDisease)
- Integration metrics (avg_bio)

**Comparison:** UCE vs MLP baseline from previous experiments

## Troubleshooting

### Issue: HuggingFace Model Download Fails

If the model fails to download from HuggingFace:
1. Check internet connectivity from compute node
2. Set HuggingFace cache: `export HF_HOME=/scratch/users/moritzs/.cache/huggingface`
3. Pre-download manually:
   ```bash
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('KuanP/uce-cosmx-geneset')"
   ```

### Issue: Submodule Not Found

If `modules/data_collection_exp` is missing:
```bash
git submodule update --init --recursive
```

### Issue: Import Errors

If you get `ModuleNotFoundError: No module named 'data_collection_exp'`:
```bash
conda activate cellwhisperer
pip install -e modules/data_collection_exp
```

### Issue: Out of Memory

If the job OOMs with UCE model:
- UCE has 773M parameters (vs MLP with <1M)
- May need to reduce batch size in delta config
- Check GPU memory usage in logs

### Issue: Gene Tokenization Errors

If you see errors about missing genes:
- Verify `static/UCE/gene_names.txt` and `static/UCE/all_species_gene_dict.json` exist
- Check `config.yaml` has correct paths under `uce_paths`

## Verification Checklist

Before declaring success, verify:

- [ ] Job completes without errors
- [ ] Model trains for 4 epochs
- [ ] W&B run shows metrics logging correctly
- [ ] Checkpoint is saved to `results/model_training/`
- [ ] Validation metrics are comparable to baseline
- [ ] No warnings about missing genes or tokenization issues

## Next Steps After Successful Run

1. Compare UCE results to MLP baseline from `SUMMARY.md`
2. Analyze metrics: retrieval, zero-shot, integration
3. Document findings in experiment summary
4. If successful, merge `uce-kuan` branch to main
5. Update documentation for future UCE usage

## Files Modified in This Branch

- `config.yaml` - Updated UCE paths, removed legacy entries
- `src/cellwhisperer/jointemb/uce_model.py` - Complete rewrite
- `src/cellwhisperer/jointemb/model.py` - Updated UCE loading (line 689-691)
- `pixi.toml` - Updated dependencies (transformers>=4.57.1, numpy<2)
- `src/shared/rules/download_models.smk` - Commented out old UCE download
- `static/UCE/` - New gene data files
- `modules/data_collection_exp/` - New submodule (Kuan's implementation)
- `src/experiments/lymphoma_trimodal_optimization/delta_config/uce.yaml` - New experiment config
- `src/experiments/lymphoma_trimodal_optimization/launch_uce.sh` - Launch script

## Commits on uce-kuan Branch

1. `e73bbb2b` - Working new UCE implementation of Kuan
2. `79d2cec7` - Move UCE resources to static/UCE and update submodule to fork
3. `d25afa00` - Remove legacy UCE code and document deprecation
4. `2f9c8c1c` - Add UCE delta config for lymphoma trimodal optimization
5. `d16fe83a` - Add launch script for UCE experiment on Sherlock
