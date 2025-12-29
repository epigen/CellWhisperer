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

# Build minimal AnnData (small X to reduce memory)
X = np.zeros((emb.shape[0], 1), dtype=np.float32)
var = pd.DataFrame(index=["dummy"]) 
adata_min = ad.AnnData(X=X, obs=obs, var=var)
adata_min.obsm["transcriptome_embeds"] = emb

# Preserve useful keys if present
for key in ["X_cellwhisperer_umap"]:
    if key in adata_b.obsm_keys():
        umap = np.asarray(adata_b.obsm[key], dtype=np.float32)
        adata_min.obsm[key] = umap

adata_min.write_h5ad(OUT_PATH, compression="lzf")
print(f"Wrote subset: {OUT_PATH} with {adata_min.n_obs} cells")
