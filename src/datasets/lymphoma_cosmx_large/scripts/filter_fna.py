"""Filter out fine needle aspirate (FNA) cores (TISS_prefix == 'SHF')."""

import anndata
import logging

logging.basicConfig(level=logging.INFO)

adata = anndata.read_h5ad(snakemake.input.adata)
logging.info(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")

mask = adata.obs["TISS_prefix"] != "SHF"
adata_filtered = adata[mask].copy()
logging.info(f"Filtered FNA: {adata.n_obs} -> {adata_filtered.n_obs} cells ({adata_filtered.n_obs/adata.n_obs*100:.1f}% retained)")

adata_filtered.write_h5ad(snakemake.output.adata)
logging.info("Done.")
