"""
Disentangle combined h5ad file into separate per-TMA h5ad files.

This script splits a single h5ad file containing multiple TMAs into
individual h5ad files, grouping TMAs according to the alignment mapping:
- TMA13 + TMA14 → TMA13_14
- TMA15 + TMA16 → TMA15_16
- TMA11 + TMA12 → TMA11_12
- TMA4-Sec1 + TMA4-Sec2 → TMA4
- Others (TMA1, TMA2, TMA3, TMA5) remain individual
"""

import anndata
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Get parameters from snakemake
combined_h5ad_path = snakemake.input.combined_h5ad
output_dir = Path(snakemake.output.output_dir)
tma_column = snakemake.params.tma_column
sample_ids = snakemake.params.sample_ids
tma_grouping = snakemake.params.tma_grouping

logging.info(f"Loading combined h5ad from {combined_h5ad_path}")
adata = anndata.read_h5ad(combined_h5ad_path)

logging.info(f"Total cells in combined dataset: {adata.n_obs}")
logging.info(f"Using TMA identifier column: {tma_column}")

# Check if the TMA column exists
if tma_column not in adata.obs.columns:
    raise ValueError(
        f"Column '{tma_column}' not found in adata.obs. Available columns: {list(adata.obs.columns)}"
    )

# Get unique TMAs in the data
unique_tmas = adata.obs[tma_column].dropna().unique()
logging.info(f"Found {len(unique_tmas)} unique TMAs in data: {sorted(unique_tmas)}")

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Process each sample (which may combine multiple individual TMAs)
for sample_id in sample_ids:
    logging.info(f"\nProcessing sample: {sample_id}")
    
    # Get the individual TMA IDs for this sample
    individual_tmas = tma_grouping[sample_id]
    logging.info(f"  Grouping TMAs: {individual_tmas}")
    
    # Filter cells for all TMAs in this group
    tma_mask = adata.obs[tma_column].isin(individual_tmas)
    n_cells = tma_mask.sum()
    
    if n_cells == 0:
        logging.warning(f"No cells found for sample {sample_id}, skipping")
        continue
    
    logging.info(f"  Found {n_cells} cells total")
    
    # Show breakdown by individual TMA
    for tma_id in individual_tmas:
        n_cells_tma = (adata.obs[tma_column] == tma_id).sum()
        logging.info(f"    {tma_id}: {n_cells_tma} cells")
    
    # Create subset AnnData
    adata_tma = adata[tma_mask].copy()
    
    # Save to output file
    output_path = output_dir / f"{sample_id}.h5ad"
    logging.info(f"  Saving to {output_path}")
    adata_tma.write_h5ad(output_path)
    logging.info(f"  Saved {sample_id}: {adata_tma.n_obs} cells, {adata_tma.n_vars} genes")

logging.info("\nDisentanglement complete!")
