"""
Randomly subsample a dataset to 1/nth of its original size.
Works on h5ad files at the cell level.
"""
import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path

# Get parameters from snakemake
input_path = snakemake.input.full_dataset
output_path = snakemake.output.subsampled_dataset
n = snakemake.params.n
seed = snakemake.params.seed

# Set random seed for reproducibility
np.random.seed(seed)

# Load the full dataset
adata = ad.read_h5ad(input_path)

# Determine the number of cells to keep
total_cells = adata.n_obs
cells_to_keep = max(1, total_cells // n)

# Randomly sample cell indices
indices_to_keep = np.random.choice(total_cells, size=cells_to_keep, replace=False)
indices_to_keep.sort()  # Keep sorted for reproducibility

# Create subsampled dataset
adata_sub = adata[indices_to_keep, :].copy()

# Ensure output directory exists
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

# Save subsampled dataset
adata_sub.write_h5ad(output_path)

print(f"Subsampled dataset from {total_cells} to {adata_sub.n_obs} cells (1/{n}th)")