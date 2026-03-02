"""
Process the LBCL atlas h5ad: group cells by annotation columns,
create pseudobulk + sampled single cells per group.
Adapted from cellxgene_census/scripts/process_dataset.py.
"""
import argparse
import anndata
import pandas as pd
import numpy as np
import json
from scipy import sparse
import scanpy as sc
import gc
import logging

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--info-json", required=True)
parser.add_argument("--min-genes-per-cell", type=int, default=100)
parser.add_argument("--n-single-cells", type=int, default=5)
parser.add_argument("--max-unique-values", type=int, default=500)
args = parser.parse_args()

IGNORE_COLS = ["soma_joinid", "donor_id", "observation_joinid"]
ALWAYS_USE_COLS = ["cell_type", "disease", "tissue", "sex", "development_stage",
                   "self_reported_ethnicity", "suspension_type"]

adata = anndata.read_h5ad(args.input)
logging.info(f"Loaded: {adata.shape}")

if adata.raw is not None:
    adata.X = adata.raw.X
    adata.raw = None

sc.pp.filter_cells(adata, min_genes=args.min_genes_per_cell)
logging.info(f"After min_genes filter: {adata.shape}")

n_unique = adata.obs.nunique()
relevant_obs_cols = n_unique[
    (n_unique < args.max_unique_values) & (n_unique < adata.n_obs)
].index.tolist()
relevant_obs_cols = [c for c in relevant_obs_cols if c not in IGNORE_COLS]
relevant_obs_cols = sorted(set(relevant_obs_cols + ALWAYS_USE_COLS))
logging.info(f"Grouping by {len(relevant_obs_cols)} columns: {relevant_obs_cols}")

unique_combos = adata.obs[relevant_obs_cols].drop_duplicates().reset_index(drop=True)
logging.info(f"Found {len(unique_combos)} unique annotation groups")

adatas_processed = []
for i, row in unique_combos.iterrows():
    conditions = [
        (adata.obs[col] == row[col]) if row[col] == row[col]
        else adata.obs[col] != adata.obs[col]
        for col in relevant_obs_cols
    ]
    mask = pd.concat(conditions, axis=1).all(axis=1)
    adata_group = adata[mask]

    if adata_group.n_obs < 1:
        continue
    if i % 100 == 0:
        logging.info(f"Group {i}/{len(unique_combos)} ({adata_group.n_obs} cells)")

    pseudobulk_X = sparse.csr_matrix(np.mean(adata_group.X, axis=0).reshape(1, -1))
    pseudobulk = anndata.AnnData(
        X=pseudobulk_X,
        obs=pd.DataFrame(adata_group.obs.iloc[0]).T[relevant_obs_cols],
        var=adata_group.var.copy(),
    )
    pseudobulk.obs.index = [f"lbcl_pseudobulk_{i}"]
    pseudobulk.obs["is_pseudobulk"] = "True"
    pseudobulk.obs["based_on_n_cells"] = adata_group.n_obs
    adatas_processed.append(pseudobulk)

    np.random.seed(42)
    n_sample = min(args.n_single_cells, adata_group.n_obs)
    idx = np.random.choice(adata_group.obs.index, size=n_sample, replace=False)
    sampled = adata_group[idx].copy()
    sampled.obs = sampled.obs[relevant_obs_cols].copy()
    sampled.obs.index = [f"lbcl_{i}_cell{x}" for x in range(n_sample)]
    sampled.obs["is_pseudobulk"] = "False"
    sampled.obs["based_on_n_cells"] = 1
    adatas_processed.append(sampled)
    gc.collect()

n_total_cells = adata.n_obs
del adata
gc.collect()

result = anndata.concat(adatas_processed)
logging.info(f"Result: {result.shape} ({len(unique_combos)} groups)")

result.uns["dataset_title"] = "Single cell atlas of large B-cell lymphoma"
result.uns["study_description"] = (
    "Large B-cell lymphomas (LBCL) are clinically and biologically diverse lymphoid "
    "malignancies with intricate microenvironments that play a key role in disease "
    "development. Single-nucleus multiome profiling on 232 tumor and control biopsies "
    "to characterize cell types and subsets in LBCL tumors, defining lymphoma "
    "microenvironment archetype profiles (LymphoMAPs)."
)
result.uns["collection_doi"] = "10.1016/j.ccell.2025.06.002"
result.uns["n_primary_cells"] = int(n_total_cells)
result.uns["obs_columns_used_for_splitting"] = relevant_obs_cols

result.obs = result.obs.infer_objects()
result.write_h5ad(args.output)

info = {
    "dataset_title": result.uns["dataset_title"],
    "n_transcriptomes": result.n_obs,
    "n_groups": len(unique_combos),
    "n_primary_cells": int(n_total_cells),
    "obs_columns": relevant_obs_cols,
}
with open(args.info_json, "w") as f:
    json.dump(info, f, indent=2)

logging.info("Done")
