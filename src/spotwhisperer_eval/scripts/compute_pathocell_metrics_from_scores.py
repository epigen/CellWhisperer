#!/usr/bin/env python3
"""
Compute PathoCell metrics across datasets from stored scores and processed AnnData.

Inputs (Snakemake):
- snakemake.input.scores: list of CSVs, one per (dataset, seed)
- snakemake.input.adatas: list of h5ad files, one per dataset (patch-level)
- snakemake.params.prediction_level: expected 'patch'

Outputs:
- snakemake.output.aggregated: JSON with aggregated (across datasets) macro and distribution metrics
- snakemake.output.per_class: CSV with per-class metrics aggregated across datasets/seeds
- snakemake.output.per_dataset: CSV with per-dataset metrics (seed-averaged)
"""

from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score


prediction_level = snakemake.params.prediction_level
score_fps = [Path(p) for p in snakemake.input.scores]
adata_fps = [Path(p) for p in snakemake.input.adatas]

out_agg = Path(snakemake.output.aggregated)
out_per_class = Path(snakemake.output.per_class)
out_per_dataset = Path(snakemake.output.per_dataset)
out_per_class_by_dataset = Path(snakemake.output.per_class_by_dataset)


def parse_dataset_from_scores(fp: Path, prediction_level: str) -> tuple[str, int]:
    stem = fp.stem
    m = re.match(rf"(.+?)_{re.escape(prediction_level)}_scores_seed(\d+)$", stem)
    if m:
        return m.group(1), int(m.group(2))
    m = re.search(r"(reg\d+_[AB])", stem)
    ds = m.group(1) if m else stem
    seed = 0
    return ds, seed


def parse_dataset_from_adata(fp: Path, prediction_level: str) -> str:
    stem = fp.stem
    suf = f"_{prediction_level}"
    return stem[:-len(suf)] if stem.endswith(suf) else stem


def soft_roc_auc(y_soft: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    pos_w = y_soft[order]
    neg_w = (1.0 - y_soft)[order]
    s_sorted = y_score[order]
    uniq, idx_start, counts = np.unique(s_sorted, return_index=True, return_counts=True)
    cum_neg = np.cumsum(neg_w)
    auc_num = 0.0
    for start, count in zip(idx_start, counts):
        end = start + count
        pos_sum = pos_w[start:end].sum()
        neg_sum = neg_w[start:end].sum()
        neg_before = cum_neg[start - 1] if start > 0 else 0.0
        auc_num += pos_sum * neg_before + 0.5 * pos_sum * neg_sum
    Wpos = pos_w.sum()
    Wneg = neg_w.sum()
    return auc_num / (Wpos * Wneg + 1e-12)


# Load adatas and cache per dataset
adata_by_ds = {}
for fp in adata_fps:
    ds = parse_dataset_from_adata(fp, prediction_level)
    adata = ad.read_h5ad(fp)
    counts_df = adata.obsm["cell_type_counts_coarse"].copy()
    classes = list(counts_df.columns)
    obs_labels = adata.obs["cell_type_coarse"].astype(str).values
    true_probs = counts_df.values.astype(float)
    true_probs = true_probs / (true_probs.sum(axis=1, keepdims=True) + 1e-12)
    adata_by_ds[ds] = {
        "classes": classes,
        "obs_labels": obs_labels,
        "true_probs": true_probs,
        "n": adata.n_obs,
    }


per_dataset_rows_raw = []  # per (dataset, seed)
per_class_rows_all = []    # accumulate across all datasets/seeds

for fp in score_fps:
    ds, seed = parse_dataset_from_scores(fp, prediction_level)
    if ds not in adata_by_ds:
        continue
    info = adata_by_ds[ds]
    classes = info["classes"]
    y_true_labels = info["obs_labels"]
    true_probs = info["true_probs"]
    sdf = pd.read_csv(fp)
    # Align score columns to class names; some files have numeric columns '0','1',...
    if list(sdf.columns) != classes:
        if str(sdf.columns[0]).isdigit():
            # Columns are positional indices; rename by order to class names
            if len(sdf.columns) >= len(classes):
                sdf = sdf.iloc[:, :len(classes)]
            sdf.columns = classes
        else:
            # Columns already named but possibly different order; select in required order
            sdf = sdf[classes]
    scores = sdf.values.astype(float)
    # Use stable softmax to obtain valid probability distributions from logits
    m = scores - scores.max(axis=1, keepdims=True)
    expm = np.exp(m)
    pred_probs = expm / (expm.sum(axis=1, keepdims=True) + 1e-12)
    ct_to_idx = {ct: i for i, ct in enumerate(classes)}
    y_true_idx = np.array([ct_to_idx[ct] for ct in y_true_labels])
    pred_top1_idx = np.argmax(scores, axis=1)

    per_class_rows_this = []
    for j, cls in enumerate(classes):
        # Hard one-vs-rest using dominant label for F1/precision/accuracy
        y_bin_hard = (y_true_idx == j).astype(int)
        s = scores[:, j]
        y_pred_bin = (pred_top1_idx == j).astype(int)
        tp = int(((y_bin_hard == 1) & (y_pred_bin == 1)).sum())
        fp = int(((y_bin_hard == 0) & (y_pred_bin == 1)).sum())
        fn = int(((y_bin_hard == 1) & (y_pred_bin == 0)).sum())
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else np.nan
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else np.nan
        accuracy = ((y_true_idx == pred_top1_idx).astype(int)).mean()
        # AUROC: use presence-of-class (from empirical distribution) for robust binary labels
        y_soft = true_probs[:, j]
        y_bin_presence = (y_soft > 0).astype(int)
        if (y_bin_presence.sum() == 0) or (y_bin_presence.sum() == y_bin_presence.size):
            rocauc = np.nan
        else:
            rocauc = roc_auc_score(y_bin_presence, s)
        # Distribution-aware (soft) AUROC
        soft_rocauc = soft_roc_auc(y_soft, s)
        # Per-class probability error metrics (MAE, MSE) over patches
        mae_prob = float(np.mean(np.abs(pred_probs[:, j] - true_probs[:, j])))
        mse_prob = float(np.mean((pred_probs[:, j] - true_probs[:, j]) ** 2))
        # Recall@5 (hard): use dominant label positives
        top5 = np.argsort(scores, axis=1)[:, -5:]
        pos_count = int((y_bin_hard == 1).sum())
        recall_at_5 = (
            ((top5 == j).any(axis=1) & (y_bin_hard == 1)).sum() / pos_count
            if pos_count > 0
            else np.nan
        )
        per_class_rows_this.append(
            {
                "dataset": ds,
                "seed": seed,
                "class_label": cls,
                "f1": f1,
                "precision": precision,
                "accuracy": accuracy,
                "rocauc": rocauc,
                "soft_rocauc": soft_rocauc,
                "mae_prob": mae_prob,
                "mse_prob": mse_prob,
                "recall_at_5": recall_at_5,
            }
        )
    per_class_df_this = pd.DataFrame(per_class_rows_this)
    per_class_rows_all.append(per_class_df_this)

    macro = {
        "f1_macroAvg": float(np.nanmean(per_class_df_this["f1"].values)),
        "precision_macroAvg": float(np.nanmean(per_class_df_this["precision"].values)),
        "accuracy_macroAvg": float(np.nanmean(per_class_df_this["accuracy"].values)),
        "rocauc_macroAvg": float(np.nanmean(per_class_df_this["rocauc"].values)),
        "soft_rocauc_macroAvg": float(np.nanmean(per_class_df_this["soft_rocauc"].values)),
        "recall_at_5_macroAvg": float(np.nanmean(per_class_df_this["recall_at_5"].values)),
    }

    cross_entropies = []
    kl_divergences = []
    js_divergences = []
    for i in range(pred_probs.shape[0]):
        pred = pred_probs[i]
        true = true_probs[i]
        ce = -(true * np.log(pred + 1e-8)).sum()
        cross_entropies.append(ce)
        kl = (true * np.log((true + 1e-8) / (pred + 1e-8))).sum()
        kl_divergences.append(kl)
        js = jensenshannon(true, pred) ** 2
        js_divergences.append(js)
    entropy_pred = np.mean([-(p * np.log(p + 1e-8)).sum() for p in pred_probs])
    entropy_true = np.mean([-(e * np.log(e + 1e-8)).sum() for e in true_probs])
    prob_corr = float(np.corrcoef(pred_probs.flatten(), true_probs.flatten())[0, 1])

    dist = {
        "mean_cross_entropy": float(np.mean(cross_entropies)),
        "std_cross_entropy": float(np.std(cross_entropies)),
        "mean_kl_divergence": float(np.mean(kl_divergences)),
        "mean_js_divergence": float(np.mean(js_divergences)),
        "entropy_predicted": float(entropy_pred),
        "entropy_empirical": float(entropy_true),
        "probability_correlation": float(prob_corr),
    }

    row = {"dataset": ds, "seed": seed}
    row.update(macro)
    row.update(dist)
    per_dataset_rows_raw.append(row)


per_dataset_df = pd.DataFrame(per_dataset_rows_raw)
if not per_dataset_df.empty:
    per_dataset_mean = per_dataset_df.groupby("dataset", as_index=False).mean(numeric_only=True)
else:
    per_dataset_mean = pd.DataFrame(columns=["dataset"])  # empty

if per_class_rows_all:
    per_class_all = pd.concat(per_class_rows_all, axis=0, ignore_index=True)
    per_class_mean = per_class_all.groupby("class_label", as_index=False).mean(numeric_only=True)
    # Per-class by dataset (seed-averaged) for error bars in per-class plots
    per_class_by_dataset = per_class_all.groupby(["dataset", "class_label"], as_index=False).mean(numeric_only=True)
else:
    per_class_mean = pd.DataFrame(columns=["class_label"])  # empty
    per_class_by_dataset = pd.DataFrame(columns=["dataset", "class_label"])  # empty

AGG_KEYS = [
    "f1_macroAvg",
    "precision_macroAvg",
    "accuracy_macroAvg",
    "rocauc_macroAvg",
    "soft_rocauc_macroAvg",
    "recall_at_5_macroAvg",
    "mean_cross_entropy",
    "std_cross_entropy",
    "mean_kl_divergence",
    "mean_js_divergence",
    "entropy_predicted",
    "entropy_empirical",
    "probability_correlation",
]

aggregated = {k: float(per_dataset_mean[k].mean()) if (k in per_dataset_mean.columns and len(per_dataset_mean)) else float("nan") for k in AGG_KEYS}
aggregated["n_datasets"] = int(per_dataset_mean.shape[0])

out_agg.parent.mkdir(parents=True, exist_ok=True)
out_per_class.parent.mkdir(parents=True, exist_ok=True)
out_per_dataset.parent.mkdir(parents=True, exist_ok=True)

with open(out_agg, "w") as f:
    json.dump(aggregated, f, indent=2)

per_class_mean.to_csv(out_per_class, index=False)
per_dataset_mean.to_csv(out_per_dataset, index=False)
per_class_by_dataset.to_csv(out_per_class_by_dataset, index=False)
