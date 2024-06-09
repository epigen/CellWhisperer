import anndata
import numpy as np


# Load anndata object
adata = anndata.read_h5ad(snakemake.input.read_count_table_full)

# Identify the unique cell types
cell_types = adata.obs["cell_ontology_class"].unique()

# For each cell type, randomly select 100 cells
np.random.seed(42)
subsampled_cells = []
for cell_type in cell_types:
    cell_indices = np.where(adata.obs["cell_ontology_class"] == cell_type)[0]
    subsampled_indices = np.random.choice(
        cell_indices, size=min(100, len(cell_indices)), replace=False
    )
    subsampled_cells.append(subsampled_indices)

# Concatenate the indices of the subsampled cells
subsampled_indices = np.hstack(subsampled_cells)

# Subset the anndata object using the subsampled indices
subsampled_adata = adata[subsampled_indices].copy()

# Save the anndata objects as an .h5ad file
subsampled_adata.write(snakemake.output.read_count_table_100_cells_per_type)

# Save structured annotations
subsampled_adata.obs.to_json(snakemake.output.structured_annotations_100percelltype)
