"""
Filter processed lymphoma CosMx data for high-quality cells only.

This script takes the output from process_data and filters for cells annotated 
as good_quality by pathologists, returning a new AnnData file with the _goodquality suffix.
"""

import anndata
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Get input and output from snakemake
input_adata_path = snakemake.input.adata
output_adata_path = snakemake.output.adata

logging.info(f"Filtering good quality cells from {input_adata_path}")
logging.info(f"Output will be saved to {output_adata_path}")

# Load the processed AnnData
adata = anndata.read_h5ad(input_adata_path)
logging.info(f"Loaded AnnData with {adata.n_obs} observations and {adata.n_vars} variables")

# Check if good_quality column exists
if "good_quality" not in adata.obs.columns:
    raise ValueError(
        f"good_quality column not found in adata.obs. Available columns: {list(adata.obs.columns)}"
    )

# Filter for good quality cells
n_cells_before = adata.n_obs
logging.info(f"good_quality column found with dtype: {adata.obs['good_quality'].dtype}")

# Handle both boolean and categorical ("yes"/"no") values
good_quality_mask = adata.obs["good_quality"]
if good_quality_mask.dtype == 'object' or good_quality_mask.dtype.name == 'category':
    # Convert "yes"/"no" to boolean
    logging.info("Converting categorical good_quality values to boolean")
    good_quality_mask = good_quality_mask.astype(str).str.lower() == "yes"
else:
    logging.info("Using boolean good_quality values directly")

# Apply filter
adata_filtered = adata[good_quality_mask].copy()
n_cells_after = adata_filtered.n_obs

logging.info(
    f"Filtered for good_quality cells: {n_cells_before} -> {n_cells_after} cells "
    f"({n_cells_after/n_cells_before*100:.1f}% retained)"
)

if n_cells_after == 0:
    logging.warning("No good quality cells found! Output will be empty.")
elif n_cells_after == n_cells_before:
    logging.info("All cells were already good quality.")

# Save the filtered AnnData
logging.info(f"Saving filtered AnnData to {output_adata_path}")
adata_filtered.write_h5ad(output_adata_path)

logging.info("Good quality filtering complete.")