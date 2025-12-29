import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch

from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.utils.inference import score_left_vs_right

# Paths
H5AD_PATH = "cellxgene.h5ad"
STATS_PATH = "results/patient_signals/stats_mannwhitney.csv"
OUT_DIR = "results/cell_level_de"
os.makedirs(OUT_DIR, exist_ok=True)

# Select target term from stats CSV (prefer 'Quiescent cells')
stats = pd.read_csv(STATS_PATH)
base_stats = stats[~stats["term"].str.startswith("ratio_")]
if (base_stats["term"] == "Quiescent cells").any():
    TARGET_TERM = "Quiescent cells"
else:
    TARGET_TERM = base_stats.sort_values("pvalue").iloc[0]["term"]

# Load data
adata = ad.read_h5ad(H5AD_PATH)

# Load model and text embedding for the target term
CKPT_PATH = "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/models/jointemb/old_format/spotwhisperer_cellxgene_census__archs4_geo.ckpt"
pl_model, tokenizer, transcriptome_processor, image_processor = (
    load_cellwhisperer_model(model_path=CKPT_PATH, eval=True)
)
model = pl_model.model
logit_scale = model.discriminator.temperature.exp()
text_embeds = model.embed_texts([TARGET_TERM], chunk_size=64)

# Score cells for target term
scores_terms_vs_cells, _ = score_left_vs_right(
    left_input=adata,
    right_input=text_embeds,
    logit_scale=logit_scale,
    model=model,
    average_mode=None,
    grouping_keys=None,
    batch_size=2,
    score_norm_method=None,
)
# Convert to per-cell Series
target_scores = pd.Series(
    scores_terms_vs_cells.T.cpu().numpy()[:, 0],
    index=adata.obs_names,
    name=f"cw_score__{TARGET_TERM}",
)

# Thresholds for high/low groups
high_thr = target_scores.quantile(0.95)
low_thr = target_scores.quantile(0.60)

# Assign groups
labels = pd.Series("mid", index=target_scores.index)
labels[target_scores > high_thr] = "high"
labels[target_scores < low_thr] = "low"

# Subset to high/low only
keep_idx = labels[labels.isin(["high", "low"])].index
adata_sub = adata[keep_idx].copy()
adata_sub.obs[f"cw_group__{TARGET_TERM}"] = labels.loc[keep_idx].astype("category")
adata_sub.obs[f"cw_score__{TARGET_TERM}"] = target_scores.loc[keep_idx].values

# Optional light preprocessing (assumes X is suitable; keep minimal)
# sc.pp.normalize_total(adata_sub)
# sc.pp.log1p(adata_sub)

# Differential expression: high vs low
sc.tl.rank_genes_groups(
    adata_sub,
    groupby=f"cw_group__{TARGET_TERM}",
    groups=["high"],
    reference="low",
    method="wilcoxon",
)

# Save DE table
de_df = sc.get.rank_genes_groups_df(adata_sub, group="high")

de_out = os.path.join(OUT_DIR, f"de_{TARGET_TERM.replace(' ', '_')}_high_vs_low.csv")
de_df.to_csv(de_out, index=False)

# Save group membership and thresholds
labels.loc[keep_idx].to_csv(
    os.path.join(OUT_DIR, f"groups_{TARGET_TERM.replace(' ', '_')}.csv"), header=True
)
with open(
    os.path.join(OUT_DIR, f"thresholds_{TARGET_TERM.replace(' ', '_')}.txt"), "w"
) as f:
    f.write(f"term: {TARGET_TERM}\n")
    f.write(f"high_thr (95th): {high_thr}\n")
    f.write(f"low_thr (60th): {low_thr}\n")
    f.write(f"n_high: {(labels=='high').sum()}\n")
    f.write(f"n_low: {(labels=='low').sum()}\n")

# Save raw gene-level counts for filtered cells (cells x genes)
counts = adata_sub.X
counts_df = pd.DataFrame(
    counts.A if hasattr(counts, "A") else counts,
    index=adata_sub.obs_names,
    columns=adata_sub.var_names,
)
counts_out = os.path.join(
    OUT_DIR, f"counts_{TARGET_TERM.replace(' ', '_')}_cells_x_genes.csv"
)
counts_df.to_csv(counts_out)

print(f"Done. Target term: {TARGET_TERM}. Outputs in {OUT_DIR}")
