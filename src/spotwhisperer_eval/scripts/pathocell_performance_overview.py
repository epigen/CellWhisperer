#!/usr/bin/env python3
"""
PathoCell performance overview plot with configurable metric (via snakemake.params.metric).

- Supports 'dataset' or 'class' aggregation via snakemake.params.scatter_unit
- Reads per-class CSVs for class-level metrics (e.g., f1, accuracy, rocauc, recall@k)
- Reads per-dataset results JSONs for patch-level distribution metrics (e.g., mean_cross_entropy)
- Fails if the requested metric is not present in the corresponding inputs
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use(snakemake.input.mpl_style)

def to_paths(obj):
    if isinstance(obj, (str, Path)):
        return [Path(obj)]
    return [Path(p) for p in obj]

# Inputs (robust to single path or list)
per_class_a = to_paths(snakemake.input.per_class_a)
per_class_b = to_paths(snakemake.input.per_class_b)
results_a = to_paths(snakemake.input.results_a)
results_b = to_paths(snakemake.input.results_b)

# Params
model_a = snakemake.params.model_a
model_b = snakemake.params.model_b
prediction_level = snakemake.params.prediction_level
scatter_unit = snakemake.params.scatter_unit
metric_col = snakemake.params.metric  # e.g., 'f1' or 'mean_cross_entropy'

# Output
out_path = Path(snakemake.output.plot)
out_path.parent.mkdir(parents=True, exist_ok=True)

JSON_METRICS = {
    "mean_cross_entropy",
    "std_cross_entropy",
    "mean_kl_divergence",
    "mean_js_divergence",
    "entropy_predicted",
    "entropy_empirical",
    "probability_correlation",
}


def load_per_class(files):
    rows = []
    for fp in files:
        df = pd.read_csv(fp)
        # normalize label col
        label_col = None
        for cand in ["class_label", "cell_type", "label", "class", "target"]:
            if cand in df.columns:
                label_col = cand
                break
        if label_col is None:
            nonnum = [c for c in df.columns if df[c].dtype == object]
            label_col = nonnum[0]
        df = df.rename(columns={label_col: "class_label"})
        # parse dataset from filename
        stem = fp.stem
        suffix_pat = r"_" + re.escape(prediction_level) + r"_per_class_seed\d+$"
        dataset = re.sub(suffix_pat, "", stem)
        if metric_col not in df.columns:
            raise KeyError(
                f"Requested metric '{metric_col}' not found in per-class file: {fp}"
            )
        # assign dataset before selecting columns
        df["dataset"] = dataset
        df = df[["class_label", "dataset", metric_col]].copy()
        rows.append(df)
    return pd.concat(rows, axis=0, ignore_index=True)


def load_results_json(files):
    rows = []
    for fp in files:
        # Support aggregated per-dataset CSV from metrics_from_scores
        if fp.suffix == ".csv":
            df = pd.read_csv(fp)
            if ("dataset" not in df.columns) or (metric_col not in df.columns):
                raise KeyError(f"Per-dataset CSV missing required columns: {fp}")
            rows.append(df[["dataset", metric_col]].copy())
        else:
            with open(fp, "r") as f:
                obj = json.load(f)
            stem = fp.stem
            suffix_pats = [
                r"_" + re.escape(prediction_level) + r"_prediction_seed\d+$",
                r"_" + re.escape(prediction_level) + r"_results_seed\d+$",
            ]
            dataset = stem
            for pat in suffix_pats:
                new = re.sub(pat, "", stem)
                if new != stem:
                    dataset = new
                    break
            if metric_col not in obj:
                raise KeyError(
                    f"Requested metric '{metric_col}' not found in results JSON: {fp}"
                )
            rows.append(pd.DataFrame({"dataset": [dataset], metric_col: [obj[metric_col]]}))
    return pd.concat(rows, axis=0, ignore_index=True)


use_json = metric_col in JSON_METRICS

if use_json:
    if prediction_level != "patch":
        raise ValueError(
            f"Metric '{metric_col}' is only available for prediction_level='patch'"
        )
    if scatter_unit == "class":
        raise ValueError(
            f"Metric '{metric_col}' is not class-level; use scatter_unit='dataset'"
        )
    a_df = load_results_json(results_a)
    b_df = load_results_json(results_b)
    # Aggregate across seeds to mean per dataset
    a_seed_mean = a_df.groupby("dataset", as_index=False)[metric_col].mean()
    b_seed_mean = b_df.groupby("dataset", as_index=False)[metric_col].mean()
else:
    a_df = load_per_class(per_class_a)
    b_df = load_per_class(per_class_b)
    # Exclude generic/background classes (if present)
    exclude_labels = {
        "Other cells",
        "Background",
        "Background cells",
        "A sample of Background cells",
    }
    a_df = a_df[~a_df["class_label"].isin(exclude_labels)].copy()
    b_df = b_df[~b_df["class_label"].isin(exclude_labels)].copy()
    # Aggregate across seeds per dataset x class
    agg_cols = ["dataset", "class_label"]
    a_seed_mean = a_df.groupby(agg_cols, as_index=False)[metric_col].mean()
    b_seed_mean = b_df.groupby(agg_cols, as_index=False)[metric_col].mean()

# Build points for scatter/violin
if scatter_unit == "dataset":
    if use_json:
        a_unit = a_seed_mean.rename(columns={metric_col: "metric_a"})
        b_unit = b_seed_mean.rename(columns={metric_col: "metric_b"})
    else:
        a_unit = (
            a_seed_mean.groupby("dataset", as_index=False)[metric_col]
            .mean()
            .rename(columns={metric_col: "metric_a"})
        )
        b_unit = (
            b_seed_mean.groupby("dataset", as_index=False)[metric_col]
            .mean()
            .rename(columns={metric_col: "metric_b"})
        )
    merged = a_unit.merge(b_unit, on="dataset", how="inner")
    label_col = "dataset"
else:
    if use_json:
        raise ValueError(
            f"Metric '{metric_col}' is not class-level; set scatter_unit='dataset'"
        )
    a_unit = (
        a_seed_mean.groupby("class_label", as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: "metric_a"})
    )
    b_unit = (
        b_seed_mean.groupby("class_label", as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: "metric_b"})
    )
    merged = a_unit.merge(b_unit, on="class_label", how="inner")
    label_col = "class_label"

# We do not have meaningful per-class p-values for JSON metrics; default False
merged["significant"] = False

# Plot
plt.figure(figsize=(2.2, 1.4))
if snakemake.params.plot_type == "violin":
    if scatter_unit == "dataset":
        if use_json:
            a_vals = a_seed_mean.set_index("dataset")[metric_col].to_numpy()
            b_vals = b_seed_mean.set_index("dataset")[metric_col].to_numpy()
        else:
            a_vals = (
                a_seed_mean.groupby("dataset", as_index=False)[metric_col]
                .mean()[metric_col]
                .to_numpy()
            )
            b_vals = (
                b_seed_mean.groupby("dataset", as_index=False)[metric_col]
                .mean()[metric_col]
                .to_numpy()
            )
    else:
        a_vals = (
            a_seed_mean.groupby("class_label", as_index=False)[metric_col]
            .mean()[metric_col]
            .to_numpy()
        )
        b_vals = (
            b_seed_mean.groupby("class_label", as_index=False)[metric_col]
            .mean()[metric_col]
            .to_numpy()
        )
    long_df = pd.DataFrame(
        {
            "model": np.repeat([model_a, model_b], [len(a_vals), len(b_vals)]),
            metric_col: np.concatenate([a_vals, b_vals]),
        }
    )
    ax = sns.violinplot(
        data=long_df,
        y="model",
        x=metric_col,
        inner="box",
        palette=["#2b7b9c", "#bfbfbf"],
        orient="h",
    )
    ax.set_xlabel(f"{metric_col} distribution")
    ax.set_yticklabels(["a", "b"])  # compact labels
    ax.set_ylabel("")
else:
    ax = sns.scatterplot(
        data=merged,
        x="metric_b",
        y="metric_a",
        hue="significant",
        palette={True: "#2b7b9c", False: "#bfbfbf"},
        s=5,
        alpha=0.4,
        # edgecolor="#0e3d4e",
        linewidth=0.4,
    )
    lims_min = float(np.nanmin([merged["metric_b"].min(), merged["metric_a"].min()]))
    lims_max = float(np.nanmax([merged["metric_b"].max(), merged["metric_a"].max()]))
    if np.isfinite(lims_min) and np.isfinite(lims_max):
        pad = 0.02 * (lims_max - lims_min + 1e-9)
        ax.plot(
            [lims_min - pad, lims_max + pad],
            [lims_min - pad, lims_max + pad],
            linestyle="--",
            color="#888888",
            linewidth=1,
        )
        # ax.set_xlim(lims_min - pad, lims_max + pad)
        # ax.set_ylim(lims_min - pad, lims_max + pad)
    ax.set_xlabel(f"{metric_col} mean ({model_b})")
    ax.set_ylabel(f"{metric_col} mean ({model_a})")
mode_label = "dataset" if scatter_unit == "dataset" else "class"
ax.set_title(
    f"{metric_col}: {model_a} vs {model_b} | {prediction_level} | by {mode_label} ({snakemake.params.plot_type})"
)
handles, labels = ax.get_legend_handles_labels()
labels = [
    "p < 0.05" if l == "True" else ("p \u2265 0.05" if l == "False" else l)
    for l in labels
]
if labels:
    ax.legend(handles, labels, title="Significance", loc="best")
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.close()
