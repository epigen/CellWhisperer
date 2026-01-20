#!/usr/bin/env python3
"""
Plot per-class macro F1 comparison: trimodal SpotWhisperer vs two baselines (CONCH, PLIP)

Inputs via Snakemake:
- snakemake.input.mpl_style: Matplotlib style file
- snakemake.input.trimodal_per_class: list of per-class CSVs for trimodal across datasets (patch-level)
- snakemake.input.adatas: list of PathoCell processed patch-level AnnData files across datasets
- snakemake.input.conch_logits: absolute path to CONCH logits CSV (terms1)
- snakemake.input.plip_logits: absolute path to PLIP logits CSV (terms1)
- snakemake.params.prediction_level: 'patch'
- snakemake.output.plot: output SVG path

Method:
- Trimodal: compute mean 'f1' per class across datasets from provided per-class CSVs
- Baselines: read logits, restrict to patch rows, map to ground truth labels using the
  processed AnnData files (join by dataset + spot_id), compute per-class F1
- Plot grouped bars per class with three methods
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad

plt.style.use(snakemake.input.mpl_style)

# Inputs
per_class_files = [Path(p) for p in snakemake.input.trimodal_per_class]
quilt_per_class_files = [Path(p) for p in snakemake.input.quilt_per_class]
adatas = [Path(p) for p in snakemake.input.adatas]
conch_fp = Path(snakemake.input.conch_logits)
plip_fp = Path(snakemake.input.plip_logits)
prediction_level = "patch"
metric = snakemake.params.metric
score_norm = snakemake.params.score_norm
out_path = Path(snakemake.output.plot)

# Normalization utility: drop leading "A sample of "


def normalize_label(s: str) -> str:
    s2 = re.sub(r"^A sample of\s+", "", s)
    # Canonicalize background/other naming
    if not s2.endswith("cells"):
        s2 = s2 + " cells"
    return s2


def apply_score_norm(mat: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return mat
    if mode == "softmax":
        m = mat - mat.max(axis=1, keepdims=True)
        expm = np.exp(m)
        return expm / (expm.sum(axis=1, keepdims=True) + 1e-9)
    if mode == "zscore":
        mu = mat.mean(axis=1, keepdims=True)
        sd = mat.std(axis=1, keepdims=True) + 1e-9
        return (mat - mu) / sd
    return mat


# --- Load trimodal per-class F1 and aggregate across datasets ---
trimodal_dfs = []
for fp in per_class_files:
    df = pd.read_csv(fp)
    label_col = None
    for cand in ["class_label", "cell_type", "label", "class", "target"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        # assume first non-numeric column is the label
        nonnum = [c for c in df.columns if df[c].dtype == object]
        label_col = nonnum[0]
    df = df.copy()
    if label_col != "class_label":
        df.rename(columns={label_col: "class_label"}, inplace=True)
    # parse dataset name from filename by stripping suffix
    stem = fp.stem  # e.g., reg006_B_patch_per_class_seed0
    suffix_pat = r"_" + re.escape(prediction_level) + r"_per_class_seed\d+$"
    dataset = re.sub(suffix_pat, "", stem)
    df["dataset"] = dataset
    trimodal_dfs.append(df)
trimodal_df = pd.concat(trimodal_dfs, axis=0, ignore_index=True)

# Load quilt1m per-class F1 and aggregate
quilt_dfs = []
for fp in quilt_per_class_files:
    df = pd.read_csv(fp)
    label_col = None
    for cand in ["class_label", "cell_type", "label", "class", "target"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        nonnum = [c for c in df.columns if df[c].dtype == object]
        label_col = nonnum[0]
    df = df.copy()
    if label_col != "class_label":
        df.rename(columns={label_col: "class_label"}, inplace=True)
    # parse dataset name from filename by stripping suffix (same as trimodal)
    stem = fp.stem
    suffix_pat = r"_" + re.escape(prediction_level) + r"_per_class_seed\d+$"
    dataset = re.sub(suffix_pat, "", stem)
    df["dataset"] = dataset
    quilt_dfs.append(df)
quilt_df = pd.concat(quilt_dfs, axis=0, ignore_index=True)
# F1 column naming in per-class CSVs
f1_col = "f1" if "f1" in trimodal_df.columns else "F1"
trimodal_agg = (
    trimodal_df.groupby(["class_label"], as_index=False)[f1_col]
    .mean()
    .rename(columns={f1_col: "trimodal"})
)

# Quilt aggregation
quilt_agg = (
    quilt_df.groupby(["class_label"], as_index=False)[f1_col]
    .mean()
    .rename(columns={f1_col: "bimodal_quilt1m"})
)
# Normalize labels to match baseline column names
trimodal_agg["class_label"] = (
    trimodal_agg["class_label"].astype(str).map(normalize_label)
)
quilt_agg["class_label"] = quilt_agg["class_label"].astype(str).map(normalize_label)

# --- Build ground truth mapping from AnnData files ---
# Map: (dataset_id, spot_id) -> true class label
truth_map = {}
for ad_fp in adatas:
    dataset_id = ad_fp.stem.replace(
        f"_{prediction_level}", ""
    )  # reg001_A, reg014_B, ...
    # load AnnData

    adata = ad.read_h5ad(ad_fp)
    # fixed label/id columns for patch-level
    label_col = "cell_type_coarse"
    id_col = "patch_id"
    # Build mapping
    if id_col is None or id_col not in adata.obs.columns:
        ids = adata.obs_names
    else:
        ids = adata.obs[id_col].astype(str).values
    labels = adata.obs[label_col].astype(str).values
    for sid, lab in zip(ids, labels):
        truth_map[(dataset_id, str(sid))] = lab


# --- Load baseline logits and compute per-class F1 ---
def load_baseline_logits(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    # keep only patch-level rows
    df = df[df["source_image"].str.contains("_patch.tiff")].copy()

    # parse dataset id and spot_id from columns
    def parse_dataset_id(x: str) -> str:
        m = re.search(r"(reg\d+_[AB])_patch\.tiff", x)
        return m.group(1) if m else x

    df["dataset_id"] = df["source_image"].apply(parse_dataset_id)
    df["spot_id"] = df["spot_id"].astype(str)
    # class columns = all except the identifiers
    id_cols = {"source_image", "spot_id", "dataset_id"}
    class_cols = [c for c in df.columns if c not in id_cols]
    corrected_classes = [normalize_label(s) for s in class_cols]
    # rename
    df.rename(
        columns={old: new for old, new in zip(class_cols, corrected_classes)},
        inplace=True,
    )
    return df, corrected_classes


conch_df, conch_classes = load_baseline_logits(conch_fp)
plip_df, plip_classes = load_baseline_logits(plip_fp)
# Use intersection with trimodal classes for consistent comparison
classes_trimodal = [
    normalize_label(s) for s in trimodal_agg["class_label"].astype(str).tolist()
]
# Exclude background/other (match behavior of existing plots)
exclude = {
    "Other cells",
    "Background cells",
    "A sample of Other cells",
    "A sample of Background cells",
}
classes = [c for c in classes_trimodal if c not in exclude]

# Helper to compute per-class metrics given logits DataFrame
from sklearn.metrics import roc_auc_score


def compute_metrics_by_class(df: pd.DataFrame, class_cols: list[str]) -> pd.DataFrame:
    # scores matrix and normalization
    scores = df[class_cols].astype(float).values
    scores = apply_score_norm(scores, mode=score_norm)

    # ground truth labels from truth_map
    y_true_raw = [
        truth_map.get((row["dataset_id"], row["spot_id"])) for _, row in df.iterrows()
    ]
    y_true = [normalize_label(str(s)) if s is not None else None for s in y_true_raw]

    # restrict to classes of interest and drop missing
    keep_idx = [i for i, t in enumerate(y_true) if (t is not None) and (t in classes)]
    if len(keep_idx) == 0:
        return pd.DataFrame({"class_label": classes, metric: [np.nan] * len(classes)})
    y_true_filt = [y_true[i] for i in keep_idx]
    scores_filt = scores[keep_idx, :]

    # compute per-class metric
    vals = []
    for j, cls in enumerate(class_cols):
        cls_norm = normalize_label(cls)
        if cls_norm not in classes:
            continue
        # one-vs-rest true labels
        y_bin = np.array([1 if t == cls_norm else 0 for t in y_true_filt])
        s = scores_filt[:, j]
        if metric == "f1":
            # argmax prediction
            pred_idx = np.argmax(scores_filt, axis=1)
            y_pred_bin = (pred_idx == j).astype(int)
            tp = int(((y_bin == 1) & (y_pred_bin == 1)).sum())
            fp = int(((y_bin == 0) & (y_pred_bin == 1)).sum())
            fn = int(((y_bin == 1) & (y_pred_bin == 0)).sum())
            denom = 2 * tp + fp + fn
            val = (2 * tp / denom) if denom > 0 else np.nan
        elif metric == "auroc":
            try:
                val = roc_auc_score(y_bin, s)
            except Exception:
                val = np.nan
        else:
            val = np.nan
        vals.append((cls_norm, val))

    # Fill missing classes with NaN
    val_map = {k: v for k, v in vals}
    out_vals = [val_map.get(cls, np.nan) for cls in classes]
    return pd.DataFrame({"class_label": classes, metric: out_vals})


conch_scores = compute_metrics_by_class(conch_df, conch_classes).rename(
    columns={metric: "conch"}
)
plip_scores = compute_metrics_by_class(plip_df, plip_classes).rename(
    columns={metric: "plip"}
)

# Merge all (include quilt1m)
merged = (
    trimodal_agg.merge(quilt_agg, on="class_label", how="inner")
    .merge(conch_scores, on="class_label", how="inner")
    .merge(plip_scores, on="class_label", how="inner")
)
# Keep classes in the chosen order
# order = [c for c in classes if c in merged["class_label"].tolist()]
# order by trimodal performance
order = merged.sort_values(by="trimodal", ascending=False)["class_label"].tolist()

merged = merged.set_index("class_label").loc[order].reset_index()

# Long-form for plotting
plot_df = merged.melt(
    id_vars=["class_label"],
    value_vars=["trimodal", "bimodal_quilt1m", "conch", "plip"],
    var_name="method",
    value_name="f1",
)

# Colors
method_colors = {
    "trimodal": "#f39c12",  # matches MODALITY_COLORS["trimodal"]
    "bimodal_quilt1m": "#16a085",
    "conch": "#2b7b9c",
    "plip": "#7f7f7f",
}

# Plot
plt.figure(figsize=(max(3.8, len(order) * 0.18), 3.2))
ax = sns.barplot(
    data=plot_df,
    x="class_label",
    y="f1",
    hue="method",
    hue_order=["trimodal", "bimodal_quilt1m", "conch", "plip"],
    order=order,
    palette=method_colors,
    edgecolor="#333333",
)
ax.set_ylabel("Macro F1 (per class)")
ax.set_xlabel("Cell type")
ax.set_title("PathoCell (patch): Trimodal vs Quilt1m vs CONCH vs PLIP")
cleaned_labels = [normalize_label(cls) for cls in order]
ax.set_xticklabels(cleaned_labels, rotation=60, ha="right")
ax.legend(title="Method")
plt.tight_layout()
plt.savefig(out_path)
plt.close()
