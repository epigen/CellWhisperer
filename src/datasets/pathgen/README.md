# PathGen-1.6M Dataset Pipeline

This pipeline processes the PathGen-1.6M dataset for use with CellWhisperer, following the same structure as the quilt1m dataset.

## Dataset Overview

PathGen-1.6M contains 1.6 million high-quality pathology image-caption pairs generated through multi-agent collaboration. The dataset uses TCGA whole-slide images with AI-generated captions describing tissue characteristics.

**Paper**: [PathGen-1.6M: 1.6 Million Pathology Image-text Pairs Generation through Multi-agent Collaboration](https://arxiv.org/abs/2407.00203)

## Prerequisites

1. **HuggingFace Dataset Access**: 
   - Visit [https://huggingface.co/datasets/jamessyx/PathGen](https://huggingface.co/datasets/jamessyx/PathGen)
   - Accept the dataset terms and conditions

2. **GDC Client Installation**:
   ```bash
   # Install GDC Data Transfer Tool
   # Follow instructions at: https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Getting_Started/
   ```

3. **Dependencies**:
   - openslide-python
   - pandas, numpy, anndata
   - PIL (Pillow)

## Pipeline Structure

The pipeline consists of the following steps:

1. **Download Metadata** (`download_pathgen_metadata`): Downloads the 1GB JSON file from HuggingFace
2. **Extract File IDs** (`extract_file_ids`): Parses JSON to extract unique GDC file IDs  
3. **Download WSI Files** (`download_gdc_file`): Downloads .svs files from TCGA via GDC
4. **Extract SVS Metadata** (`extract_svs_metadata`): Diagnostic extraction of technical specifications
5. **Filter Metadata** (`filter_pathgen_metadata`): Filters to only include available WSI files
6. **Create H5AD Files** (`create_pathgen_h5ads`): Generates h5ad files with center-oriented coordinates

## Usage

### Full Pipeline
```bash
cd src/datasets/pathgen
pixi run --no-progress snakemake --cores 1
```

### Testing Mode
To run with only 2 WSIs for testing, update `config.yaml`:
```yaml
pathgen_testing: true
```

Then run:
```bash
pixi run --no-progress snakemake --cores 1
```

### SVS Diagnostic Mode
To extract and view technical specifications from downloaded SVS files:
```bash
# Extract metadata from all downloaded SVS files
pixi run --no-progress snakemake diagnose_svs

# Or run just the metadata extraction
pixi run --no-progress snakemake extract_svs_metadata
```

### Manual Steps Required

1. **Download the JSON metadata manually**:
   ```bash
   mkdir -p resources/pathgen
   # After accepting HuggingFace terms:
   wget https://huggingface.co/datasets/jamessyx/PathGen/resolve/main/PathGen-1.6M.json \
        -O resources/pathgen/pathgen_1.6m.json
   ```

2. **Install GDC client** if not already available:
   - Follow [GDC installation guide](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Getting_Started/)

## Output Structure

The pipeline creates:

- `results/pathgen/h5ads/`: Directory containing individual h5ad files for each WSI
- `results/pathgen/pathgen_filtered.json`: Filtered metadata
- `results/pathgen/svs_metadata.json`: Technical specifications of SVS files (JSON format)
- `results/pathgen/svs_metadata_report.txt`: Human-readable diagnostic report
- `results/pathgen/svs_metadata_report.csv`: Summary statistics in CSV format
- `results/pathgen/test_patch_*.png`: Example patches for quality control

Each h5ad file contains:
- **obs**: Patch metadata with center coordinates (`x_pixel`, `y_pixel`), `natural_language_annotation` (captions), WSI IDs
- **obsm**: Spatial coordinates (center-oriented)
- **uns**: Image metadata including `image_path`, `coordinate_system="center"`, `pixel_size=0.25`
- **X**: Empty matrix (following quilt1m structure)

## Coordinate System

PathGen uses **center-oriented coordinates**:
- `x_pixel`, `y_pixel`: Center coordinates of 672×672 patches
- Physical patch size: 168 μm × 168 μm (at 0.25 μm/pixel)
- Use coordinate utilities for OpenSlide conversion when needed

## Technical Specifications

Based on TCGA whole slide images:
- **Magnification**: 40x objective power
- **Pixel Size**: 0.25 μm/pixel (typical)
- **Patch Size**: 672×672 pixels = 168 μm × 168 μm
- **File Format**: Aperio SVS (proprietary tiled TIFF)
- **Coordinate System**: Center-oriented

## Configuration

Key parameters in `config.yaml`:

```yaml
pathgen_config:
  patch_size: 224  # Use 224x224 patches at 40x magnification (0.25 μm/pixel)
  max_patches_per_wsi: 50  # Limit patches per WSI

pathgen_testing: false  # Set to true for testing with 2 WSIs

dataset_he_mapping:
  pathgen: detailed_resolution  # Use 40x magnification (0.25 μm/pixel)
```

## File Structure

```
src/datasets/pathgen/
├── Snakefile                    # Main pipeline definition
├── README.md                    # This file
└── scripts/
    ├── extract_file_ids.py      # Extract GDC file IDs from JSON
    ├── filter_metadata.py       # Filter metadata by available files  
    ├── extract_svs_metadata.py  # Extract SVS technical specifications
    ├── coordinate_utils.py       # Center coordinate utilities
    └── create_h5ads.py          # Create h5ad files with center coordinates
```

## Working with Coordinates

Use the provided utility functions for coordinate conversions:

```python
from scripts.coordinate_utils import center_to_topleft, read_patch_from_center, get_physical_coordinates

# Convert center coordinates to top-left for OpenSlide
topleft_x, topleft_y = center_to_topleft(center_x, center_y, patch_size=224)

# Read patch directly from center coordinates
patch = read_patch_from_center(slide, center_x, center_y)

# Convert to physical coordinates
x_um, y_um = get_physical_coordinates(center_x, center_y, pixel_size=0.25)
```

## Data Format

Each entry in the PathGen JSON follows this structure:
```json
{
    "wsi_id": "TCGA-AA-3844-01Z-00-DX1.bf88ce1f-0601-40c8-813e-4e3df51bd2f0",
    "position": ["35136", "33344"],
    "caption": "Detailed pathology description...",
    "file_id": "bffacf34-4942-496d-9c5d-d36294d80a9d"
}
```

The resulting h5ad files maintain compatibility with the existing CellWhisperer training pipeline.

## Troubleshooting

1. **JSON Download Issues**: Ensure you've accepted HuggingFace dataset terms
2. **GDC Download Failures**: Check internet connection and GDC client installation
3. **OpenSlide Errors**: Ensure OpenSlide is properly installed with WSI format support
4. **Memory Issues**: Reduce `max_patches_per_wsi` in testing mode

## Testing

Use testing mode to validate the pipeline with a small subset:

```bash
# Edit config.yaml to set pathgen_testing: true
pixi run --no-progress snakemake test_pipeline
```

This will process only 2 WSIs with maximum 10 patches each.