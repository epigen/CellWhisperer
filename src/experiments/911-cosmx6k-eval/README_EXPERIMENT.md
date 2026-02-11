# CosMx6K Evaluation Experiment (911)

This experiment evaluates different architectures and approaches for the CosMx6K dataset (lymphoma_cosmx_large) against the baseline configuration.

## Overview

The CosMx6K dataset contains ~6000 cells from 11 TMAs (8 files due to joint scanning) from lymphoma tissue. This experiment tests:
- **CNN architecture** improvements (ramped up capacity)
- **MPNN architecture** (graph-based spatial relationships)
- **UUL finetuning** (training the transcriptome and text models)
- **Data quality filtering** (excluding low-quality TMAs)
- **Context vs cell-level** information

## Base Configuration

`base_config.yaml` - Contains the baseline configuration with best practices from cosmx1k-eval:
- **Dataset**: `lymphoma_cosmx_large` + `cellxgene_census`
- **Image encoder**: UNI2 with ramped-up CNN (512 dim, 4 layers, cell_level_model enabled)
- **Transcriptome model**: MLP (random initialization)
- **Text model**: BERT (locked)
- **Locking mode**: uLL (Unlock transcriptome with random init, Lock text, Lock image)
- **Batch size**: 1024
- **Learning rate**: 0.001

## Best Configuration

`best_config.yaml` - Alternative configuration testing UUL locking mode:
- **Same as base config** but with UUL locking mode (Unlock text, Unlock transcriptome, Lock image)
- Tests whether fine-tuning text model helps (vs uLL baseline which only unlocks transcriptome)
- All other settings identical to base_config.yaml
- Note: This is different from finetune_geneformer_config which uses Geneformer instead of MLP

## Delta Configurations

Each delta configuration tests a specific hypothesis by modifying the base config:

### 1. `standard_cnn_config.yaml`
**Purpose**: Test whether ramped-up CNN capacity is necessary (baseline uses ramped CNN)
**Changes**: 
- Reduces CNN embedding dimension from 512 to 128
- Reduces number of CNN layers from 4 to 2
**Expected Effect**: Tests if smaller CNN is sufficient or if ramped-up capacity helps
**WandB run name**: `cosmx6k-standard-cnn`

### 2. `disable_cell_level_config.yaml`
**Purpose**: Test importance of context vs cell-level information
**Changes**: Disables cell-level encoder (`image_config.cell_level_model: false`)
**Expected Effect**: Tests whether cell-level features are important vs context-only features
**WandB run name**: `cosmx6k-no-cell-level`

### 3. `cell_cnn_only_config.yaml`
**Purpose**: Test importance of context model by using only cell-level features
**Changes**: 
- Disables context-level encoder
- Uses ramped-up CNN for cell-level features only
**Expected Effect**: Tests the value of context information
**WandB run name**: `cosmx6k-cell-cnn-only`

### 4. `mpnn_config.yaml`
**Purpose**: Test graph-based architecture for spatial relationships
**Changes**: 
- Uses MPNN (Message Passing Neural Network) instead of UNI2
- 512-dim hidden layers, 4 layers, mean aggregation
**Expected Effect**: May capture spatial relationships better than CNN
**WandB run name**: `cosmx6k-mpnn`

### 5. `finetune_geneformer_config.yaml`
**Purpose**: Fine-tune Geneformer with UUL locking mode
**Changes**: 
- Changes locking mode to UUL (Unlock text, Unlock transcriptome, Lock image)
- Reduces learning rate to 0.0001 for fine-tuning regime
**Expected Effect**: Allows fine-tuning of both text and transcriptome models while keeping image frozen
**WandB run name**: `cosmx6k-finetune-uul`

### 6. `finetune_uni_config.yaml`
**Purpose**: Fine-tune UNI2 image encoder with LUU locking mode
**Changes**: 
- Changes locking mode to LUU (Lock text, Unlock transcriptome, Unlock image)
- Reduces learning rate to 0.0001 for fine-tuning regime
**Expected Effect**: Allows fine-tuning of transcriptome and image models while keeping text frozen
**WandB run name**: `cosmx6k-finetune-luu`

### 7. `geneformer_transcriptome_config.yaml`
**Purpose**: Test pretrained Geneformer vs MLP (baseline uses MLP)
**Changes**: 
- Changes transcriptome model from MLP to Geneformer
- Reduces batch size to 512 (Geneformer is larger)
- Keeps uLL locking mode (random init for fair comparison)
**Expected Effect**: Tests if pretrained model helps vs simple MLP
**WandB run name**: `cosmx6k-geneformer`

### 8. `lul_locking_config.yaml`
**Purpose**: Test LUL locking mode (unlock via projection instead of random init)
**Changes**: 
- Changes locking mode to LUL (vs uLL in baseline)
- This uses pretrained projection layer instead of random initialization
**Expected Effect**: Tests pretrained projection vs random initialization
**WandB run name**: `cosmx6k-lul`

### 9. `good_quality_tmas_config.yaml`
**Purpose**: Test impact of filtering low-quality TMAs
**Changes**: Uses dataset built with `DROP_LOW_QUALITY_TMAS=True` (excludes TMA1 and TMA11_12)
**Expected Effect**: Cleaner data by excluding TMAs marked as poor quality
**WandB run name**: `cosmx6k-good-quality-tmas`
**Note**: Requires dataset to be built with quality filtering enabled in Snakefile

## Running Experiments

### Option 1: Launch all jobs via SLURM

```bash
cd src/experiments/911-cosmx6k-eval
./launch_cosmx6k_eval.sh
```

This will submit 10 jobs total:
- 1 baseline job (MLP + uLL + ramped CNN)
- 9 delta config jobs

### Option 2: Run individual experiments

```bash
# Run baseline
cellwhisperer fit --config base_config.yaml

# Run specific delta config
cellwhisperer fit --config base_config.yaml --config delta_config/ramp_up_cnn_config.yaml

# Run best config
cellwhisperer fit --config best_config.yaml
```

## Dataset Requirements

The experiment uses the `lymphoma_cosmx_large` dataset. Ensure it is properly built:

```bash
cd src/datasets/lymphoma_cosmx_large
pixi run --no-progress snakemake --cores 8
```

For the good quality TMA filtering experiment, rebuild with:
```bash
# Edit Snakefile to set DROP_LOW_QUALITY_TMAS = True
pixi run --no-progress snakemake --cores 8
```

## Monitoring

All jobs are tracked in WandB:
- **Project**: SpatialWhisperer
- **Entity**: single-cellm
- **Group**: 911-cosmx6keval

Monitor SLURM jobs:
```bash
squeue -u $USER
```

Check logs:
```bash
ls -ltr slurm_logs/
tail -f slurm_logs/slurm-<job_id>-<job_name>.out
```

## Research Questions

This experiment aims to answer:

1. **Architecture comparison**: Does CNN, MPNN, or UUL perform best on CosMx6K data?
2. **Context importance**: How important is context vs cell-level information?
3. **Transcriptome model**: Does MLP outperform Geneformer for limited gene panels?
4. **Fine-tuning**: Does fine-tuning pretrained models help?
5. **Data quality**: Does filtering low-quality TMAs improve performance?
6. **Cross-holdout correlation**: Does performance across different holdout sets correlate?
7. **Scale effects**: How does performance scale from CosMx1K to CosMx6K?

## Expected Outcomes

Based on insights from the cosmx1k-eval experiment (918), we expect:
- Ramped-up CNN to improve performance
- Context model to be important (cell-only may underperform)
- MLP transcriptome model may work better than Geneformer for CosMx data
- Quality filtering may improve metrics

## Implementation Status

All configurations are ready to run:
- ✅ Base configuration (Geneformer + LUL)
- ✅ Best configuration (MLP + UUL + ramped CNN)
- ✅ CNN architecture variants
- ✅ MPNN architecture
- ✅ Finetuning variants (UUL, LUU)
- ✅ MLP transcriptome model
- ✅ Quality filtering config
- ✅ Launch script

## Next Steps

After running experiments:
1. Analyze results in WandB
2. Compare performance metrics across configurations
3. Identify best-performing architecture
4. Document insights for future experiments
5. Consider metric improvements (e.g., retrieval AUROC within single cores)
