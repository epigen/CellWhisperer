#!/usr/bin/env python3
"""
Compute lung baseline metrics from per-sample score CSVs and h5ad files.

The lung h5ad files use 'region_type_expert_annotation' with values
NOR, TUM, TLS, INFL (and UNASSIGNED which is excluded).
These map to the TERMS_1 class names used in the baseline CSVs:
  NOR  -> 'normal cells'
  TUM  -> 'tumor cells'
  TLS  -> 'tertiary lymphoid structure'
  INFL -> 'infiltrating cells'

To match SpotWhisperer's evaluation (which filters out NOR, INFL, UNASSIGNED),
we restrict to TUM and TLS only. The score CSVs may contain all 4 class columns;
we subset to the evaluated classes and recompute argmax over the 2-class scores.

Inputs (Snakemake):
- snakemake.input.scores: list of per-sample score CSVs
- snakemake.input.adatas: list of h5ad files (one per sample)

Outputs:
- snakemake.output.aggregated: JSON with aggregated metrics
- snakemake.output.per_class: CSV with per-class metrics (mean across samples)
- snakemake.output.per_dataset: CSV with per-sample metrics
- snakemake.output.per_class_by_dataset: CSV with per-class metrics per sample
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import roc_auc_score

# Matches SW's filter_classes (keeps TUM + TLS, drops NOR, INFL, UNASSIGNED)
ANNOTATION_TO_CLASS = {
    "TUM": "tumor cells",
    "TLS": "tertiary lymphoid structure",
}
CLASSES = list(ANNOTATION_TO_CLASS.values())

score_fps = [Path(p) for p in snakemake.input.scores]
adata_fps = [Path(p) for p in snakemake.input.adatas]


def sample_id_from_adata_fp(fp: Path) -> str:
    # LC1.h5ad -> lc_1
    stem = fp.stem  # e.g. 'LC1'
    n = stem[2:]    # '1'
    return f"lc_{n}"


def sample_id_from_score_fp(fp: Path) -> str:
    # lc_1_scores_seed0.csv -> lc_1
    return fp.stem.replace("_scores_seed0", "")


# Load adatas indexed by sample id (lc_1 etc.)
# Store the boolean mask so we can apply the same filter to score CSVs
adata_by_sample = {}
for fp in adata_fps:
    sid = sample_id_from_adata_fp(fp)
    adata = ad.read_h5ad(fp)
    ann = adata.obs["region_type_expert_annotation"].astype(str)
    mask = ann.isin(ANNOTATION_TO_CLASS.keys())
    adata_by_sample[sid] = {
        "labels": ann[mask].map(ANNOTATION_TO_CLASS).values,
        "mask": mask.values,  # boolean array, True for non-UNASSIGNED spots
        "n_total": adata.n_obs,
    }

per_dataset_rows = []
per_class_rows_all = []

for fp in score_fps:
    sid = sample_id_from_score_fp(fp)
    if sid not in adata_by_sample:
        continue
    info = adata_by_sample[sid]
    y_true_labels = info["labels"]  # string class names, UNASSIGNED excluded
    mask = info["mask"]

    sdf = pd.read_csv(fp)
    # Subset to evaluated classes only (score CSV may have extra class columns)
    sdf = sdf[CLASSES]

    # Score CSV has one row per spot in the h5ad (including UNASSIGNED).
    # Apply the same boolean mask to keep only annotated spots.
    assert len(sdf) == info["n_total"], f"Score CSV rows ({len(sdf)}) != h5ad spots ({info['n_total']}) for {sid}"
    sdf = sdf.loc[mask].reset_index(drop=True)

    scores = sdf.values.astype(float)
    pred_top1_idx = np.argmax(scores, axis=1)
    y_true_idx = np.array([CLASSES.index(c) for c in y_true_labels])

    per_class_rows_this = []
    for j, cls in enumerate(CLASSES):
        y_bin = (y_true_idx == j).astype(int)
        s = scores[:, j]
        y_pred_bin = (pred_top1_idx == j).astype(int)
        tp = int(((y_bin == 1) & (y_pred_bin == 1)).sum())
        fp_ = int(((y_bin == 0) & (y_pred_bin == 1)).sum())
        fn = int(((y_bin == 1) & (y_pred_bin == 0)).sum())
        denom = 2 * tp + fp_ + fn
        f1 = (2 * tp / denom) if denom > 0 else np.nan
        precision = (tp / (tp + fp_)) if (tp + fp_) > 0 else np.nan
        accuracy = float((y_true_idx == pred_top1_idx).mean())
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            rocauc = np.nan
        else:
            rocauc = roc_auc_score(y_bin, s)
        per_class_rows_this.append({
            "dataset": sid,
            "seed": 0,
            "class_label": cls,
            "f1": f1,
            "precision": precision,
            "accuracy": accuracy,
            "rocauc": rocauc,
        })

    per_class_df_this = pd.DataFrame(per_class_rows_this)
    per_class_rows_all.append(per_class_df_this)

    macro = {k: float(np.nanmean(per_class_df_this[k].values))
             for k in ["f1", "precision", "accuracy", "rocauc"]}
    per_dataset_rows.append({"dataset": sid, "seed": 0, **macro})

per_dataset_df = pd.DataFrame(per_dataset_rows)
per_dataset_mean = per_dataset_df.groupby("dataset", as_index=False).mean(numeric_only=True) if not per_dataset_df.empty else per_dataset_df

if per_class_rows_all:
    per_class_all = pd.concat(per_class_rows_all, ignore_index=True)
    per_class_mean = per_class_all.groupby("class_label", as_index=False).mean(numeric_only=True)
    per_class_by_dataset = per_class_all.groupby(["dataset", "class_label"], as_index=False).mean(numeric_only=True)
else:
    per_class_mean = pd.DataFrame(columns=["class_label"])
    per_class_by_dataset = pd.DataFrame(columns=["dataset", "class_label"])

AGG_KEYS = ["f1", "precision", "accuracy", "rocauc"]
aggregated = {k: float(per_dataset_mean[k].mean()) if (k in per_dataset_mean.columns and len(per_dataset_mean)) else float("nan") for k in AGG_KEYS}
aggregated["n_samples"] = int(per_dataset_mean.shape[0])

with open(snakemake.output.aggregated, "w") as f:
    json.dump(aggregated, f, indent=2)
per_class_mean.to_csv(snakemake.output.per_class, index=False)
per_dataset_mean.to_csv(snakemake.output.per_dataset, index=False)
per_class_by_dataset.to_csv(snakemake.output.per_class_by_dataset, index=False)
