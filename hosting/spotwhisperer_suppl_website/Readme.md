# Spot level H&E annotation benchmark

This benchmark dataset consists of hematoxylin and eosin (H&E) images from five lung cancer samples ([Dawo et al., 2025](https://doi.org/10.5281/zenodo.14620362)), annotated with cell types and pathologically relevant labels at spot-level resolution. For methodology, refer to the accompanying publication ([Schaefer et al., 2025](https://www.biorxiv.org/content/10.1101/2025.07.14.664402v1)).

## Dataset Files

The dataset contains the following compressed h5ad files:

- LC1.h5ad.gz - Lung cancer sample 1
- LC2.h5ad.gz - Lung cancer sample 2
- LC3.h5ad.gz - Lung cancer sample 3
- LC4.h5ad.gz - Lung cancer sample 4
- LC5.h5ad.gz - Lung cancer sample 5

## Usage Instructions

### Loading and Basic Analysis

```python
import scanpy as sc

# Read the curated file
file_path = 'LC1.h5ad.gz'
adata = sc.read_h5ad(file_path)

# Display basic information
print(adata)

# Information about spots
print(adata.obs)

# Information about genes
print(adata.var)

# High-resolution image
print (adata.uns["20x_slide"])

# Sample metadata
print (adata.uns["meta"])

# Spatial coordinates
print (adata.obsm["X_spatial"])

```

## Key Data Components

Below is a summary of the essential data components for the curated data:

### Gene Expression Data

- **`adata.X`** contains raw gene expression counts (SPOTS × GENES)


### Gene Information

- **`adata.var`** contains human-readable gene names, gene IDs, and feature_types 


### Spatial Information

- **Spot coordinates:**
    - `adata.obs['x_array']`, `adata.obs['y_array']`: Original spot coordinates on the slide space
    - `adata.obs['x_pixel']`, `adata.obs['y_pixel']`: Spot coordinates in the image space
    - `adata.obsm['X_spatial']`: 2D spatial coordinates for visualization

### High-resolution image
- **Image:**
    - `adata.uns["20x_slide"]`: High-resolution image
    - `adata.uns["meta"]["magnification"]`: Magnification of the image


### Annotations and Metadata

- **Tissue coverage:**
    - `adata.obs['in_tissue']`: Boolean indicating spots within tissue boundaries as annotated by histopathologists
- **Expert annotations:**
    - `adata.obs['region_type_expert_annotation']`: Manual tissue region annotations
    - Categories: UNASSIGNED, NOR (Normal), TUM (Tumor), TLS (Tertiary Lymphoid Structures), INFL (Inflammatory)
- **Cell type annotations:**
    - `adata.obs['cell_type_annotations']`: Automated cell type predictions from reference atlas
    - Provides broad cell type classifications (e.g., Epithelial, Endothelial, Immune, Stromal)
- **Sample metadata:**
    - `adata.obs['sample_ID']`: Sample identifiers
    - `adata.obs['barcode']`: Unique spot barcodes
    - `adata.uns['meta']`: Sample metadata
    - `adata.uns['meta']['spot_diameter_fullres']`: Spot diameter
    - `adata.uns['meta']['dot_size']`: Number of subspots within a spot (DeepSpot inference parameter)


## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{Schaefer2025.07.14.664402,
  author = {Schaefer, Moritz and Nonchev, Kalin and Awasthi, Animesh and Burton, Jake and Koelzer, Viktor H and R{\\"a}tsch, Gunnar and Bock, Christoph},
  title = {Molecularly informed analysis of histopathology images using natural language},
  elocation-id = {2025.07.14.664402},
  year = {2025},
  doi = {10.1101/2025.07.14.664402},
  publisher = {Cold Spring Harbor Laboratory},
  url = {https://www.biorxiv.org/content/early/2025/07/18/2025.07.14.664402},
  eprint = {https://www.biorxiv.org/content/early/2025/07/18/2025.07.14.664402.full.pdf},
  journal = {bioRxiv}
}
```

and

```bibtex
@misc{dawo2025visium,
  title = {10x Visium spatial transcriptomics dataset: Kidney (3) and lung (5) cancer with tertiary lymphoid structures},
  author = {Dawo, Sebastian and Nonchev, Kalin and Silina, Karina},
  publisher = {Zenodo},
  year = {2025},
  url = {http://dx.doi.org/10.5281/zenodo.14620362},
  doi = {10.5281/ZENODO.14620362}
}
```
