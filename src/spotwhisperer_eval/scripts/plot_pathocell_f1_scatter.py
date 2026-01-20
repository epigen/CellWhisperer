#!/usr/bin/env python3
"""
Scatterplot of F1 scores between two models.
- Mode 'class': each point is a cell type (mean across datasets)
- Mode 'dataset': each point is a dataset (mean across cell types)
- Each point: one cell type (class_label)
- X: model_b (assumed bimodal/quilt1m if passed that way)
- Y: model_a (assumed trimodal if passed that way)
Inputs: per-class CSVs for both models (across datasets/seeds)
"""

import re
from pathlib import Path

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
prediction_level = snakemake.params.prediction_level

scatter_unit = snakemake.params.scatter_unit

# Output
out_path = Path(snakemake.output.plot)
out_path.parent.mkdir(parents=True, exist_ok=True)


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
        df["dataset"] = dataset
        rows.append(df[["class_label", "dataset", "f1"]])
    return pd.concat(rows, axis=0, ignore_index=True)


a_df = load_per_class(per_class_a)
b_df = load_per_class(per_class_b)

# Exclude generic/background classes
exclude_labels = {"Other cells", "Background cells"}
a_df = a_df[~a_df["class_label"].isin(exclude_labels)].copy()
b_df = b_df[~b_df["class_label"].isin(exclude_labels)].copy()

# Aggregate across seeds per dataset x class
agg_cols = ["dataset", "class_label"]
a_seed_mean = a_df.groupby(agg_cols, as_index=False)["f1"].mean()
b_seed_mean = b_df.groupby(agg_cols, as_index=False)["f1"].mean()

if scatter_unit == "dataset":
    # Each point = dataset average across classes
    a_unit = (
        a_seed_mean.groupby("dataset", as_index=False)["f1"]
        .mean()
        .rename(columns={"f1": "f1_a"})
    )
    b_unit = (
        b_seed_mean.groupby("dataset", as_index=False)["f1"]
        .mean()
        .rename(columns={"f1": "f1_b"})
    )
    merged = a_unit.merge(b_unit, on="dataset", how="inner")
    label_col = "dataset"
else:
    # Each point = class average across datasets
    a_unit = (
        a_seed_mean.groupby("class_label", as_index=False)["f1"]
        .mean()
        .rename(columns={"f1": "f1_a"})
    )
    b_unit = (
        b_seed_mean.groupby("class_label", as_index=False)["f1"]
        .mean()
        .rename(columns={"f1": "f1_b"})
    )
    merged = a_unit.merge(b_unit, on="class_label", how="inner")
    label_col = "class_label"

# Bring in per-class p-values only for class-level mode
if scatter_unit == "class":
    comp_base = Path(per_class_a[0]).parent.parent
    comp_csv = (
        comp_base
        / "comparison"
        / prediction_level
        / f"{model_a}_vs_{model_b}_per_class.csv"
    )  # independent of cell_type_level in path
    try:
        comp_df = pd.read_csv(comp_csv)
        p_map = dict(
            zip(
                comp_df[comp_df["metric"] == "f1"]["class_label"],
                comp_df[comp_df["metric"] == "f1"]["p_value"],
            )
        )
        merged["p_value"] = merged[label_col].map(p_map)
        merged["significant"] = merged["p_value"] < 0.05
    except Exception:
        merged["significant"] = False
else:
    merged["significant"] = False

# Plot
plt.figure(figsize=(2, 1.3))
if snakemake.params.plot_type == "violin":
    # Two violins: F1 distributions for model_a and model_b
    # Compose long-form DataFrame
    if scatter_unit == "dataset":
        a_vals = (
            a_seed_mean.groupby("dataset", as_index=False)["f1"].mean()["f1"].values
        )
        b_vals = (
            b_seed_mean.groupby("dataset", as_index=False)["f1"].mean()["f1"].values
        )
    else:
        a_vals = (
            a_seed_mean.groupby("class_label", as_index=False)["f1"].mean()["f1"].values
        )
        b_vals = (
            b_seed_mean.groupby("class_label", as_index=False)["f1"].mean()["f1"].values
        )
    long_df = pd.DataFrame(
        {
            "model": np.repeat([model_a, model_b], [len(a_vals), len(b_vals)]),
            "f1": np.concatenate([a_vals, b_vals]),
        }
    )
    ax = sns.violinplot(
        data=long_df,
        y="model",
        x="f1",
        inner="box",
        palette=["#2b7b9c", "#bfbfbf"],
        orient="h",
    )
    ax.set_xlim(0, 0.2)
    ax.set_xlabel("F1 distribution")
    ax.set_yticklabels(["a", "b"])
    ax.set_ylabel("")
else:
    ax = sns.scatterplot(
        data=merged,
        x="f1_b",
        y="f1_a",
        hue="significant",
        palette={True: "#2b7b9c", False: "#bfbfbf"},
        s=50,
        edgecolor="#0e3d4e",
        linewidth=0.4,
    )
    lims_min = float(np.nanmin([merged["f1_b"].min(), merged["f1_a"].min(), 0.0]))
    lims_max = float(np.nanmax([merged["f1_b"].max(), merged["f1_a"].max(), 1.0]))
    ax.plot(
        [lims_min, lims_max],
        [lims_min, lims_max],
        linestyle="--",
        color="#888888",
        linewidth=1,
    )
    ax.set_xlim(0, 0.38)
    ax.set_ylim(0, 0.38)
    try:
        diffs = (merged["f1_a"] - merged["f1_b"]).to_numpy()
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(diffs[~np.isnan(diffs)])
        t = np.linspace(lims_min, lims_max, 200)
        dens = kde(np.zeros_like(t))
        dens_norm = (dens - dens.min()) / (dens.ptp() + 1e-8)
        band = 0.01 * (lims_max - lims_min)
        ax.fill_between(
            t,
            t - band * dens_norm,
            t + band * dens_norm,
            color="#a0c7d7",
            alpha=0.35,
            zorder=0,
            label="diagonal density",
        )
    except Exception:
        pass
    ax.set_xlabel(f"F1 mean ({model_b})")
    ax.set_ylabel(f"F1 mean ({model_a})")
mode_label = "dataset" if scatter_unit == "dataset" else "class"
ax.set_title(
    f"F1: {model_b} vs {model_a} | {prediction_level} | by {mode_label} ({snakemake.params.plot_type})"
)
handles, labels = ax.get_legend_handles_labels()
labels = [
    "p < 0.05" if l == "True" else ("p ≥ 0.05" if l == "False" else l) for l in labels
]
if labels:
    ax.legend(handles, labels, title="Significance", loc="best")
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.close()
