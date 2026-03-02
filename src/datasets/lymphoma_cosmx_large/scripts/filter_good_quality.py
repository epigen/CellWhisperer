"""
Filter processed lymphoma CosMx large data for valid cores only.

Uses the 'valid_core' column (int, 0/1) which is a per-core binary annotation.
"""

import anndata
import logging

logging.basicConfig(level=logging.INFO)

adata = anndata.read_h5ad(snakemake.input.adata)
logging.info(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")

mask = adata.obs["valid_core"].astype(bool)
adata_filtered = adata[mask].copy()
logging.info(f"Filtered: {adata.n_obs} -> {adata_filtered.n_obs} cells ({adata_filtered.n_obs/adata.n_obs*100:.1f}% retained)")

adata_filtered.write_h5ad(snakemake.output.adata)
logging.info("Done.")