# CosMx1K Evaluation Experiment (918)

This experiment evaluates different approaches to improve CosMx1K performance against the baseline configuration.

## Base Configuration

`base_config.yaml` - Contains the baseline configuration against which all variants are tested.

## Delta Configurations

Each delta configuration represents a specific experimental variant that modifies only certain parameters from the base config:

### 1. `disable_cell_level_config.yaml`
**Purpose**: Test importance of context vs cell-level information
**Changes**: Disables cell-level encoder (`image_config.cell_level_model: false`)
**Implementation Status**: ✅ Ready to run
**Expected Effect**: Tests whether cell-level features are important vs context-only features

### 2. `ramp_up_cnn_config.yaml`
**Purpose**: Improve CNN architecture with higher capacity
**Changes**: 
- Increases CNN embedding dimension from 128 to 512
- Increases number of CNN layers from 2 to 4
**Implementation Status**: ✅ Ready to run
**Expected Effect**: Better representation capacity for image features



### 3. `finetune_geneformer_config.yaml`
**Purpose**: CosMx1K vs whole transcriptome issue (option 2a: fine-tune Geneformer with UUL)
**Changes**: 
- Changes locking mode to UUL (Unlock text, Unlock transcriptome, Lock image)
- Reduces learning rate for fine-tuning regime
**Implementation Status**: ✅ Ready to run
**Expected Effect**: Allows fine-tuning of both text and transcriptome models while keeping image frozen

### 3b. `finetune_geneformer_luu_config.yaml`
**Purpose**: CosMx1K vs whole transcriptome issue (option 2b: fine-tune Geneformer with LUU)
**Changes**: 
- Changes locking mode to LUU (Lock text, Unlock transcriptome, Unlock image)
- Reduces learning rate for fine-tuning regime
**Implementation Status**: ✅ Ready to run
**Expected Effect**: Allows fine-tuning of transcriptome and image models while keeping text frozen

### 4. `improved_alignment_config.yaml`
**Purpose**: Test importance of pixel-perfect H&E alignment
**Changes**: Uses core-aligned datasets (lymphoma_cosmx_small_singlecell_corealigned_*)
**Implementation Status**: ✅ Ready to run
**Expected Effect**: Better spatial correspondence between H&E and transcriptome data

### 5. `good_quality_cells_config.yaml`
**Purpose**: Test impact of using only pathologist-annotated good quality cells
**Changes**: Uses datasets with `filter_good_quality=True` (lymphoma_cosmx_small_singlecell_goodquality_*)
**Implementation Status**: ✅ Ready to run
**Expected Effect**: Cleaner data by excluding cells marked as poor quality by pathologists

## Running Experiments

Each config can be tested against the baseline by using the delta configuration system. For example:

```bash
# Run baseline
python train.py config=base_config.yaml

# Run cell-level disabled variant
python train.py config=base_config.yaml config+=delta_config/disable_cell_level_config.yaml
```

## Implementation Requirements

## Implementation Status

### Ready to Run Immediately

The following configs can be tested immediately without additional implementation:
- **`disable_cell_level_config.yaml`** - Disable cell-level encoder
- **`ramp_up_cnn_config.yaml`** - Improved CNN architecture with configurable parameters
- **`finetune_geneformer_config.yaml`** - Fine-tune with UUL locking mode
- **`finetune_geneformer_luu_config.yaml`** - Fine-tune with LUU locking mode
- **`improved_alignment_config.yaml`** - Core-aligned datasets with optimized H&E alignment
- **`good_quality_cells_config.yaml`** - Use only pathologist-annotated good quality cells

All configs are now ready to run! The core-level alignment has been implemented with inner parallelization and minimization of HSV values to find optimal tissue alignment.