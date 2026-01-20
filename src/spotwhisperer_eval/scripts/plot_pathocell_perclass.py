#!/usr/bin/env python3
"""
Create per-class boxplots of delta scores between two models across datasets.

- x: cell type (class_label)
- y: delta score = model_a - model_b, aggregated per dataset
- One figure per metric (provided via snakemake.params.metric)
- Sort classes by median delta
- Overlay faint underlined datapoints per dataset (legend label: datasets)

Assumes input per-class CSVs exist for both models with a class label column and
numeric metric columns (e.g., auroc, f1, accuracy, precision, recall@5).
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use(snakemake.input.mpl_style)

# Inputs
per_class_a = [Path(p) for p in snakemake.input.per_class_a]
per_class_b = [Path(p) for p in snakemake.input.per_class_b]

# Params
model_a = snakemake.params.model_a
model_b = snakemake.params.model_b
metric = snakemake.params.metric
# Map friendly metric names to per-class CSV column names
metric_col = {
    "auroc": "rocauc",
    "rocauc": "rocauc",
    "recall@1": "recall_at_1",
    "recall@5": "recall_at_5",
    "recall@10": "recall_at_10",
    "recall@50": "recall_at_50",
}.get(metric, metric)
prediction_level = snakemake.params.prediction_level

# Output
out_path = Path(snakemake.output.plot)
out_path.parent.mkdir(parents=True, exist_ok=True)


def load_per_class(files):
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        label_col = None
        for cand in ["class_label", "cell_type", "label", "class", "target"]:
            if cand in df.columns:
                label_col = cand
                break
        if label_col is None:
            # assume the first non-numeric column is the label
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
        df["source_file"] = fp.name
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True)


a_df = load_per_class(per_class_a)
b_df = load_per_class(per_class_b)

# Aggregate across seeds: mean per dataset x class
if metric_col not in a_df.columns or metric_col not in b_df.columns:
    raise KeyError(f"Metric '{metric}' not found in per-class files")

agg_cols = ["dataset", "class_label"]
a_g = (
    a_df.groupby(agg_cols, as_index=False)[metric_col]
    .mean()
    .rename(columns={metric_col: "a"})
)
b_g = (
    b_df.groupby(agg_cols, as_index=False)[metric_col]
    .mean()
    .rename(columns={metric_col: "b"})
)

merged = a_g.merge(b_g, on=agg_cols, how="inner")
merged["delta"] = merged["a"] - merged["b"]
exclude_labels = {"A sample of Other cells", "A sample of Background cells"}
merged = merged[~merged["class_label"].isin(exclude_labels)].copy()
# Aggregate stats per class for barplot with uncertainty
agg = merged.groupby("class_label")["delta"].agg(["mean", "std", "count"]).reset_index()
agg["se"] = agg["std"] / np.sqrt(agg["count"])

# Sort classes by mean delta (descending: highest first)
order = agg.sort_values("mean", ascending=False)["class_label"].tolist()

# Plot mean bar with uncertainty (SE)
plt.figure(figsize=(max(3.8, len(order) * 0.15), 3))
# Read p-values before plotting for coloring
comp_base = Path(per_class_a[0]).parent.parent  # PATHOCELL_RESULTS
comp_csv = (
    comp_base
    / "comparison"
    / prediction_level
    / f"{model_a}_vs_{model_b}_per_class.csv"
)
comp_df = pd.read_csv(comp_csv)
metric_in_csv = metric_col
comp_sub = comp_df[comp_df["metric"] == metric_in_csv]
p_by_class = dict(zip(comp_sub["class_label"], comp_sub["p_value"]))

# Determine bar colors based on significance
is_sig = [(p_by_class.get(cls, np.nan) < 0.05) for cls in order]
bar_colors = ["#2b7b9c" if sig else "#bfbfbf" for sig in is_sig]

ax = sns.barplot(
    data=agg,
    x="class_label",
    y="mean",
    order=order,
    palette=bar_colors,
    edgecolor="#3a647c",
)
# Add error bars (standard error)
ax.errorbar(
    x=np.arange(len(order)),
    y=agg.set_index("class_label").loc[order, "mean"].values,
    yerr=agg.set_index("class_label").loc[order, "se"].values,
    fmt="none",
    ecolor="#1f1f1f",
    elinewidth=1.2,
    capsize=2.5,
    capthick=1.2,
    zorder=3,
)
# Add a legend for significance coloring
from matplotlib.patches import Patch

legend_handles = [
    Patch(facecolor="#2b7b9c", edgecolor="#3a647c", label="p < 0.05"),
    Patch(facecolor="#bfbfbf", edgecolor="#3a647c", label="p ≥ 0.05"),
]
ax.legend(handles=legend_handles, title="Significance", loc="best")
plt.axhline(0, color="#444444", linewidth=1, linestyle="--", alpha=0.6)
plt.ylabel(f"Mean delta {metric}")  # ({model_a} - {model_b})")
plt.xlabel("Cell type")
plt.title(f"Per-class mean delta for {metric} | {prediction_level}")
# Clean display labels: drop leading 'A sample of '
cleaned_labels = [re.sub(r"^A sample of\s+", "", cls) for cls in order]
ax.set_xticklabels(cleaned_labels, rotation=60, ha="right")
plt.tight_layout()

# Annotate p-values from comparison CSV produced by pathocell_compare_models
# Derive CSV path relative to PATHOCELL_RESULTS via one of the input files
# comp_base = Path(per_class_a[0]).parent.parent  # PATHOCELL_RESULTS
# comp_csv = (
#     comp_base
#     / "comparison"
#     / prediction_level
#     / f"{model_a}_vs_{model_b}_per_class.csv"
# )
# comp_df = pd.read_csv(comp_csv)
# metric_in_csv = metric_col
# comp_sub = comp_df[comp_df["metric"] == metric_in_csv]
# p_by_class = dict(zip(comp_sub["class_label"], comp_sub["p_value"]))
# for i, cls in enumerate(order):
#     p = p_by_class.get(cls)
#     if p is not None:
#         y = agg.set_index("class_label").loc[cls, "mean"]
#         err = agg.set_index("class_label").loc[cls, "se"]
#         y_text = (
#             y
#             + (err if np.isfinite(err) else 0)
#             + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
#         )
#         ax.text(i, y_text, f"p={p:.2e}", ha="center", va="bottom", fontsize=8)

plt.savefig(out_path, dpi=200)
plt.close()
