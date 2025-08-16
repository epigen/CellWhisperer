# STHELAR Dataset Integration

## Overview

STHELAR is a spatial transcriptomics dataset with paired histology images from the S-BIAD2146 biological data archive. This implementation:

1. Downloads STHELAR data from the EBI FTP server (separate download step)
2. Processes each slide's zarr file and H&E patches to be compatible with UNIProcessor requirements (separate processing step)
3. Creates multiple adata files following the `full_dataset_multi` pattern
4. Aggregates single-cell gene expression to patch level for spatial analysis

## Data Sources

The dataset is downloaded from:
- **Main archive**: https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/146/S-BIAD2146/
- **H&E patches**: `Files/STHELAR/data_20x/data/images.zip` (15GB)
- **Gene expression**: `Files/STHELAR/sdata_slides/sdata_*.zarr.zip` files (varies by slide)

## Requirements

- Python 3.10+
- SpatialData and Scverse ecosystem libraries
- Sufficient storage space (>500GB total for all slides)
- Reliable internet connection for large downloads

## Usage

Run from the repository root:

```bash
snakemake --cores 4 --conda-frontend mamba -s src/datasets/sthelar/Snakefile
```

## Configuration

The dataset processes these STHELAR slides by default:
- 31 slides across various tissues: bone_marrow, brain, breast, colon, etc.
- Each slide contains H&E patches and single-cell spatial transcriptomics data

To modify the slide list, edit `STHELAR_SLIDE_IDS` in the Snakefile.

## Data Processing

The processing is split into two separate steps:

### 1. Data Download (`scripts/download_data.py`)
- Downloads `images.zip` containing all H&E patches (shared across slides)
- Downloads individual `sdata_*.zarr.zip` files for each slide's gene expression data
- Caches data locally to avoid re-downloads
- Uses EBI FTP server with direct HTTP downloads

### 2. Data Processing (`scripts/process_data.py`) 
Transforms STHELAR SpatialData to include:
- `x_pixel`, `y_pixel`: Spatial coordinates from patch centroids
- `20x_slide`: Representative H&E patch image as numpy array (H,W,3)
- `spot_diameter_fullres`: Patch diameter for extraction (default: 256)
- `gene_name`: Gene identifiers in var DataFrame
- `counts`: Aggregated count matrix from single cells to patches

### Key Processing Steps:
1. Load SpatialData zarr file using spatialdata library
2. Extract H&E patch coordinates from `he_patches` shapes
3. Map single cells to patches using spatial boundaries
4. Aggregate gene expression from cells within each patch
5. Create AnnData object with patch-level data

### Multi-file Output (full_dataset_multi)
- Individual slide files: `results/sthelar/full_data_{slide_id}.h5ad`

## Testing

The Snakemake workflow can be tested:

```bash
cd src/datasets/sthelar
snakemake --cores 4 --conda-frontend mamba -n  # dry run
```

## Integration

The dataset is integrated into the main CellWhisperer config:
- Added to `config.yaml` datasets list
- Uses standard path patterns for consistency  
- Compatible with existing processing pipelines

## Tissue Coverage

The STHELAR dataset includes slides from these tissue types:
- Bone marrow (2 slides)
- Brain (1 slide) 
- Breast (4 slides)
- Cervix (1 slide)
- Colon (2 slides)
- Heart (1 slide)
- Kidney (2 slides)
- Liver (2 slides)
- Lung (2 slides)
- Lymph node (1 slide)
- Ovary (2 slides)
- Pancreas (3 slides)
- Prostate (1 slide)
- Skin (4 slides)
- Tonsil (2 slides)
- Bone (1 slide)