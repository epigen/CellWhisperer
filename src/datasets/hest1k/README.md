# HEST-1K Dataset Integration

This directory contains the Snakemake workflow for downloading and processing the HEST-1K dataset for SpotWhisperer.

## Overview

HEST-1K is a spatial transcriptomics dataset with paired histology images from [MahmoodLab](https://huggingface.co/datasets/MahmoodLab/hest). This implementation:

1. Downloads HEST data from HuggingFace using the HEST library (separate download step)
2. Processes each sample to be compatible with UNIProcessor requirements (separate processing step)
3. Creates multiple adata files following the `full_dataset_multi` pattern
4. Ensures proper spatial coordinate mapping for histology patch extraction

## Requirements

- HuggingFace token with access to HEST dataset: https://huggingface.co/datasets/MahmoodLab/hest
- HEST library (automatically installed via `envs/main.yaml`)
- Access to the HEST dataset (requires accepting terms of use)

## Usage

Run from the repository root:

```bash
export HUGGINGFACE_TOKEN="your_token_here"
snakemake --cores 4 --conda-frontend mamba -s src/datasets/hest1k/Snakefile
```

## Configuration

The dataset processes these HEST-1K samples by default:
- TENX95, TENX99, TENX105, TENX108, TENX109

To modify the sample list, edit `HEST_SAMPLE_IDS` in the Snakefile.

## Data Processing

The processing is split into two separate steps:

### 1. Data Download (`scripts/download_data.py`)
- Uses `datasets.load_dataset('MahmoodLab/hest')` with sample-specific patterns
- Caches data locally to avoid re-downloads
- Supports both individual samples and downloading all samples (when `sample_ids=None`)

### 2. Data Processing (`scripts/process_data.py`) 
Transforms HEST data to include:
- `x_pixel`, `y_pixel`: Spatial coordinates from `pxl_col_in_fullres`, `pxl_row_in_fullres`
- `20x_slide`: Histology image as numpy array (H,W,3)
- `spot_diameter_fullres`: Patch diameter for extraction (default: 100)
- `gene_name`: Gene identifiers in var DataFrame
- `counts`: Raw count matrix in layers

### Multi-file Output (full_dataset_multi)
- Individual sample files: `results/hest1k/full_data_{sample_id}.h5ad`

## Testing

The Snakemake workflow can be tested:

```bash
cd src/datasets/hest1k
snakemake --cores 4 --conda-frontend mamba -n  # dry run
```

## Fallback Behavior

When HEST library is unavailable or HuggingFace access fails, the processing script creates compatible fallback data to ensure pipeline continuity.

## Integration

The dataset is integrated into the main SpotWhisperer config:
- Added to `config.yaml` datasets list
- Uses standard path patterns for consistency
- Compatible with existing processing pipelines