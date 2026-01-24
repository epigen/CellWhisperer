#!/usr/bin/env python3
"""
Violin plots of per-dataset metric deltas (model_a - model_b), one violin per metric.

Inputs via Snakemake:
- snakemake.input.mpl_style: Matplotlib style file
- snakemake.input.a_per_dataset: CSV with per-dataset metrics for model_a (seed-averaged)
- snakemake.input.b_per_dataset: CSV with per-dataset metrics for model_b (seed-averaged)
- snakemake.params.model_a, model_b, prediction_level
- snakemake.output.plot: output SVG path

The per-dataset CSVs come from pathocell_metrics_from_scores and include columns like:
f1_macroAvg, precision_macroAvg, accuracy_macroAvg, rocauc_macroAvg,
soft_rocauc_macroAvg, recall_at_5_macroAvg, mean_cross_entropy, std_cross_entropy,
mean_kl_divergence, mean_js_divergence, entropy_predicted, entropy_empirical,
probability_correlation
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use(snakemake.input.mpl_style)

a_fp = Path(snakemake.input.a_per_dataset)
b_fp = Path(snakemake.input.b_per_dataset)
out_path = Path(snakemake.output.plot)
out_path.parent.mkdir(parents=True, exist_ok=True)

model_a = snakemake.params.model_a
model_b = snakemake.params.model_b
prediction_level = snakemake.params.prediction_level

METRICS = [
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

# Load per-dataset CSVs
a_df = pd.read_csv(a_fp)
b_df = pd.read_csv(b_fp)

# Ensure common datasets
common = sorted(set(a_df.get("dataset", [])).intersection(set(b_df.get("dataset", []))))
if common:
    a_df = a_df[a_df["dataset"].isin(common)].copy()
    b_df = b_df[b_df["dataset"].isin(common)].copy()

# Compute deltas per dataset for all available metrics
rows = []
for m in METRICS:
    if (m in a_df.columns) and (m in b_df.columns):
        merged = a_df[["dataset", m]].merge(b_df[["dataset", m]], on="dataset", suffixes=("_a", "_b"))
        for _, r in merged.iterrows():
            rows.append({"dataset": r["dataset"], "metric": m, "delta": float(r[f"{m}_a"]) - float(r[f"{m}_b"])})

long_df = pd.DataFrame(rows)

# Plot: one subplot per metric, each with its own y-axis range
metrics_present = [m for m in METRICS if m in long_df["metric"].unique()]
n = len(metrics_present)
fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(9, max(1.6 * n, 3.0)), sharex=False)
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])
for ax, m in zip(axes, metrics_present):
    dfm = long_df[long_df["metric"] == m].copy()
    # Use a constant x label for a single vertical violin
    dfm["metric_label"] = m
    sns.violinplot(data=dfm, x="metric_label", y="delta", inner="box", color="#bfbfbf", ax=ax)
    ax.axhline(0, linestyle="--", color="#666666", linewidth=1)
    ax.set_title(m)
    ax.set_xlabel("")
    ax.set_ylabel(f"Δ {model_a}-{model_b}")
    ax.tick_params(axis="x", labelbottom=False)
fig.suptitle(f"PathoCell {prediction_level}: Per-dataset metric deltas", y=0.995, fontsize=10)
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.close()
