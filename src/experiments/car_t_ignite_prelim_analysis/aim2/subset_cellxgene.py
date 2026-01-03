import anndata as ad
import numpy as np
import pandas as pd
import os

IN_PATH = "cellxgene.h5ad"
OUT_PATH = "cellxgene_B_Product_lowburden.light.h5ad"

# Read without loading X into memory
adata_b = ad.read_h5ad(IN_PATH, backed="r")

# Build a single final mask to avoid view-of-view on backed AnnData
obs = adata_b.obs
mask_tp = obs["timepoint"] == "B_Product"
# Compute patient burden on B_Product only
patient_burden = obs[mask_tp].groupby("patient_id")["tumor_burden_SPD"].first()
# 80th percentile among non-missing burden
threshold = patient_burden.dropna().quantile(0.80)
# keep patients with burden <= threshold or missing burden
keep_patients = patient_burden[(patient_burden <= threshold) | (patient_burden.isna())].index
mask_pat = obs["patient_id"].isin(keep_patients)
final_mask = mask_tp & mask_pat

# Subset once
adata_b = adata_b[final_mask]

# Extract embeddings and obs
emb = np.asarray(adata_b.obsm["transcriptome_embeds"], dtype=np.float32)
obs = adata_b.obs.copy()

# Materialize to memory to carry counts and var
adata_b = adata_b.to_memory(copy=True)
# Ensure float32 embedding array is preserved
emb = np.asarray(adata_b.obsm["transcriptome_embeds"], dtype=np.float32)
adata_b.obsm["transcriptome_embeds"] = emb

# Create counts layer by inverting log1p in X
X = adata_b.X
arr = X.A if hasattr(X, "A") else X
base = None
if "log1p" in adata_b.uns and isinstance(adata_b.uns["log1p"], dict):
    base = adata_b.uns["log1p"].get("base", None)
if base is None or base == "e" or base == np.e:
    counts = np.expm1(arr)
else:
    counts = np.power(base, arr) - 1
adata_b.layers["counts"] = counts.astype(np.float32)

# Preserve useful keys if present
for key in ["X_cellwhisperer_umap"]:
    if key in adata_b.obsm_keys():
        umap = np.asarray(adata_b.obsm[key], dtype=np.float32)
        adata_b.obsm[key] = umap

adata_b.write_h5ad(OUT_PATH, compression="lzf")
print(f"Wrote subset: {OUT_PATH} with {adata_b.n_obs} cells")
