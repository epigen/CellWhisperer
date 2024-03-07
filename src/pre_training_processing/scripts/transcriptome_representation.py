import scanpy as sc
import numpy as np

adata = sc.read_h5ad(snakemake.input.read_count_table)

# simply normalize the transcriptome data (10.000 counts per cell/sample -> log1p)
sc.pp.normalize_total(adata, target_sum=10000, exclude_highly_expressed=True)
sc.pp.log1p(adata)

# Compute PCA of the data
sc.pp.pca(adata, n_comps=min(snakemake.params.n_dimensions, len(adata) - 1))

# compute HVGs (epxected log data)
# sc.pp.highly_variable_genes(adata, n_top_genes=2048)

np.savez(
    snakemake.output.representation,
    representation=adata.obsm["X_pca"],
    orig_ids=adata.obs.index.to_numpy(),
)
