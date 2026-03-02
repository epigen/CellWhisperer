#!/usr/bin/env python3
"""
Per-class metrics comparison table and bar plot for lung tissue:
SpotWhisperer models vs PLIP/CONCH baselines.

SpotWhisperer prediction CSVs (region_type_expert_annotation.by_cell.csv) have columns:
  score_for_<class>, predicted_labels, label
and only cover 2 classes (TUM, TLS) on LC1.

Baseline per-class CSVs come from lung_metrics_from_scores (5 samples, 4 classes).

The comparison is restricted to the intersection of classes: TUM and TLS
(mapped to 'tumor cells' and 'tertiary lymphoid structure').
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

plt.style.use(snakemake.input.mpl_style)

metric = snakemake.params.metric

# Label mapping: SW uses short codes, baselines use full names
SW_LABEL_MAP = {
    "TUM": "tumor cells",
    "TLS": "tertiary lymphoid structure",
}
SHARED_CLASSES = list(SW_LABEL_MAP.values())


# --- helpers ---

def read_per_class(fp: Path, metric: str) -> pd.Series:
    """Read per-class CSV from lung_metrics_from_scores and return Series indexed by class_label."""
    df = pd.read_csv(fp)
    label_col = next(c for c in ["class_label", "class", "label"] if c in df.columns)
    df = df.rename(columns={label_col: "class_label"})
    return df.set_index("class_label")[metric]


def compute_sw_per_class(fp: Path, metric: str) -> pd.Series:
    """
    Compute per-class metrics from a SpotWhisperer prediction CSV.
    Columns: score_for_<class>, predicted_labels, label (short codes).
    Returns Series indexed by full class names (SHARED_CLASSES only).
    """
    df = pd.read_csv(fp, index_col=0)
    # Map short labels to full names; drop rows with unmapped labels
    df = df[df["label"].isin(SW_LABEL_MAP)].copy()
    df["label_full"] = df["label"].map(SW_LABEL_MAP)
    df["predicted_full"] = df["predicted_labels"].map(SW_LABEL_MAP)

    # Build score column mapping: find the actual column for each class,
    # tolerating minor variants (e.g. "structures" vs "structure")
    score_col_for_class = {}
    score_columns = [c for c in df.columns if c.startswith("score_for_")]
    for cls in SHARED_CLASSES:
        # Exact match first
        exact = f"score_for_{cls}"
        if exact in df.columns:
            score_col_for_class[cls] = exact
            continue
        # Fuzzy: find columns that start with the class name (handles plural/singular)
        matches = [c for c in score_columns if cls in c.lower() or c.lower().replace("score_for_", "") in cls]
        assert len(matches) == 1, f"Expected 1 score column matching '{cls}', found {matches}"
        score_col_for_class[cls] = matches[0]

    results = {}
    y_true_idx = np.array([SHARED_CLASSES.index(c) for c in df["label_full"]])
    pred_idx = np.array([SHARED_CLASSES.index(c) if c in SHARED_CLASSES else -1
                         for c in df["predicted_full"]])
    scores_arr = np.column_stack([df[score_col_for_class[cls]].values for cls in SHARED_CLASSES])

    for j, cls in enumerate(SHARED_CLASSES):
        y_bin = (y_true_idx == j).astype(int)
        y_pred_bin = (pred_idx == j).astype(int)
        s = scores_arr[:, j]
        if metric in ("rocauc", "auroc"):
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                results[cls] = np.nan
            else:
                results[cls] = roc_auc_score(y_bin, s)
        elif metric == "f1":
            tp = int(((y_bin == 1) & (y_pred_bin == 1)).sum())
            fp_ = int(((y_bin == 0) & (y_pred_bin == 1)).sum())
            fn = int(((y_bin == 1) & (y_pred_bin == 0)).sum())
            denom = 2 * tp + fp_ + fn
            results[cls] = (2 * tp / denom) if denom > 0 else np.nan
        elif metric in ("accuracy", "precision"):
            tp = int(((y_bin == 1) & (y_pred_bin == 1)).sum())
            fp_ = int(((y_bin == 0) & (y_pred_bin == 1)).sum())
            results[cls] = float((y_true_idx == pred_idx).mean()) if metric == "accuracy" else (tp / (tp + fp_) if (tp + fp_) > 0 else np.nan)
        else:
            results[cls] = np.nan
    return pd.Series(results)


# --- load all methods ---

methods = {}

# SpotWhisperer models (prediction CSVs)
sw_inputs = {k: Path(v) for k, v in snakemake.input.items() if k.startswith("sw_")}
for key, fp in sw_inputs.items():
    label = key[3:]  # strip "sw_" prefix
    methods[label] = compute_sw_per_class(fp, metric)

# Baseline per-class CSVs (from lung_metrics_from_scores)
baseline_inputs = {k: Path(v) for k, v in snakemake.input.items() if k.startswith("baseline_")}
for key, fp in baseline_inputs.items():
    label = key[9:]  # strip "baseline_" prefix
    methods[label] = read_per_class(fp, metric)

# Restrict all to shared classes
rows = []
for method, series in methods.items():
    for cls in SHARED_CLASSES:
        rows.append({"method": method, "class_label": cls,
                     "value": series.get(cls, np.nan)})
plot_df = pd.DataFrame(rows)


# --- bar plot ---

method_order = list(methods.keys())
palette = sns.color_palette("tab10", n_colors=len(method_order))
color_map = dict(zip(method_order, palette))

fig, ax = plt.subplots(figsize=(max(4, 1.5 * len(SHARED_CLASSES)), 3.0))
sns.barplot(
    data=plot_df,
    x="class_label",
    y="value",
    hue="method",
    hue_order=method_order,
    palette=color_map,
    edgecolor="#333333",
    ax=ax,
)
ax.set_title(f"Lung tissue – {metric} per class")
ax.tick_params(axis="x", rotation=30)
ax.set_xlabel("")
ax.set_ylabel(metric)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title="Method", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
plt.tight_layout()
plt.savefig(snakemake.output.plot)
plt.close()


# --- CSV table: rows = methods, columns = classes + mean ---

table = pd.DataFrame(
    {method: {cls: series.get(cls, np.nan) for cls in SHARED_CLASSES}
     for method, series in methods.items()}
).T
table.index.name = "method"
table["mean"] = table[SHARED_CLASSES].mean(axis=1)
table.to_csv(snakemake.output.csv_table)
