"""
Replace SCTransform corrected counts in adata.X with raw read counts from CosMx flat files.

The processed h5ads have SCTransform corrected counts in X (matching nCount_SCT).
This script replaces X with the actual raw counts from the instrument flat files
(matching nCount_RNA), keeping all other fields (obs, obsm, uns, layers) intact.
"""

import anndata
import pandas as pd
import numpy as np
import scipy.sparse as sp
import logging

logging.basicConfig(level=logging.INFO)

adata = anndata.read_h5ad(snakemake.input.adata)
logging.info(f"Loaded h5ad: {adata.n_obs} cells, {adata.n_vars} genes")

raw = pd.read_csv(snakemake.input.raw_expr)
logging.info(f"Loaded raw expression CSV: {raw.shape[0]} cells, {raw.shape[1] - 2} genes")

# Parse cell_id (format: c_{slide}_{fov}_{cellID}) to extract fov and cellID
parsed = adata.obs["cell_id"].str.extract(r"c_\d+_(\d+)_(\d+)")
adata.obs["_fov_match"] = parsed[0].astype(int)
adata.obs["_cellID_match"] = parsed[1].astype(int)

# Create a merge key for both datasets
adata.obs["_merge_key"] = adata.obs["_fov_match"].astype(str) + "_" + adata.obs["_cellID_match"].astype(str)
raw["_merge_key"] = raw["fov"].astype(str) + "_" + raw["cell_ID"].astype(str)

# Identify shared genes (h5ad genes present in the CSV)
h5ad_genes = list(adata.var_names)
csv_genes = set(raw.columns) - {"fov", "cell_ID", "_merge_key"}
shared_genes = [g for g in h5ad_genes if g in csv_genes]
logging.info(f"Shared genes: {len(shared_genes)}/{len(h5ad_genes)} (missing in CSV will be zeroed)")

# Index raw by merge key, keep only shared genes
raw_sub = raw.set_index("_merge_key")[shared_genes]
# Drop any duplicate keys (shouldn't happen, but be safe)
raw_sub = raw_sub[~raw_sub.index.duplicated(keep="first")]

# Reindex to match h5ad cell order — unmatched cells get NaN (-> 0)
raw_aligned = raw_sub.reindex(adata.obs["_merge_key"].values)
n_matched = raw_aligned.notna().any(axis=1).sum()
logging.info(f"Cell matching: {n_matched}/{adata.n_obs} h5ad cells found in raw CSV")

raw_aligned = raw_aligned.fillna(0).values.astype(np.float32)

# Build full gene matrix (shared genes filled, rest zeros)
new_X = np.zeros((adata.n_obs, adata.n_vars), dtype=np.float32)
gene_to_idx = {g: i for i, g in enumerate(h5ad_genes)}
shared_idx = [gene_to_idx[g] for g in shared_genes]
new_X[:, shared_idx] = raw_aligned

new_X_sparse = sp.csr_matrix(new_X)

# Verify totals roughly match nCount_RNA
raw_totals = np.array(new_X_sparse.sum(axis=1)).flatten()
rna_totals = adata.obs["nCount_RNA"].values.astype(float)
matched_mask = raw_totals > 0
corr = np.corrcoef(raw_totals[matched_mask], rna_totals[matched_mask])[0, 1]
logging.info(f"Correlation of new X totals with nCount_RNA: {corr:.4f}")

# Replace X
adata.X = new_X_sparse

# Clean up temp columns
adata.obs.drop(columns=["_fov_match", "_cellID_match", "_merge_key"], inplace=True)

adata.write_h5ad(snakemake.output.adata)
logging.info(f"Saved raw-count h5ad: {adata.n_obs} cells, {adata.n_vars} genes")
