"""Merge natural language annotations into the final h5ad."""
import argparse
import anndata
import pandas as pd
import json
import logging

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--annotations", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

adata = anndata.read_h5ad(args.input)

with open(args.annotations) as f:
    annotations = json.load(f)

# Map annotations to obs index
mapped = adata.obs.index.map(lambda x: annotations.get(x))
adata.obs["natural_language_annotation"] = [v[0] if v else None for v in mapped]

# Store replicates if available
max_reps = max((len(v) for v in annotations.values()), default=1)
if max_reps > 1:
    rep_data = [v[1:] if v and len(v) > 1 else [] for v in mapped]
    rep_df = pd.DataFrame(
        data=rep_data,
        index=adata.obs.index,
        columns=[str(i) for i in range(1, max_reps)],
    )
    adata.obsm["natural_language_annotation_replicates"] = rep_df

# Remove unannotated cells
n_before = adata.n_obs
mask = adata.obs["natural_language_annotation"].notna()
if not mask.all():
    adata = adata[mask].copy()
    logging.warning(f"Removed {n_before - adata.n_obs} unannotated cells")

logging.info(f"Final dataset: {adata.shape}")
adata.write_h5ad(args.output)
