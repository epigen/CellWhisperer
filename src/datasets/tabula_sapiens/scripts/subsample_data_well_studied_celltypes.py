import anndata
import numpy as np

# Load anndata object
adata = anndata.read_h5ad(snakemake.input.read_count_table_full)

# Subsample to 'well-studied' cell types
subsampled_adata = adata[adata.obs["cell_ontology_class"].str.lower().isin(snakemake.params.well_studied_celltypes), :].copy()
    
# Save the anndata objects as an .h5ad file
subsampled_adata.write(snakemake.output.read_count_table_well_studied_celltypes)

# Save structured annotations
subsampled_adata.obs.to_json(snakemake.output.structured_annotations_well_studied_celltypes)

