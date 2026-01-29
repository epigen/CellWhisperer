#!/usr/bin/env python3
"""
Plot a single per-class metric comparison (one axis) for BiBridge vs Quilt1m vs CONCH variants
using metrics_from_scores per-class CSVs.

Inputs via Snakemake:
- snakemake.input.mpl_style: Matplotlib style file
- snakemake.input.*_per_class: per-class CSVs per method (seed/dataset aggregated)
- snakemake.params.metric: the metric to plot (e.g., 'rocauc')
- snakemake.output.plot: output SVG path
- snakemake.output.csv_table: output CSV with a table for the selected metric
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use(snakemake.input.mpl_style)

bibridge_pc = Path(snakemake.input.bibridge_per_class)
quilt_pc = Path(snakemake.input.quilt_per_class)
conch_LLL_pc = Path(snakemake.input.conch_LLL_per_class)
conch_LUL_pc = Path(snakemake.input.conch_LUL_per_class)
conch_frozen_pc = Path(snakemake.input.conch_frozen_per_class)
conch_terms1_pc = Path(snakemake.input.conch_terms1_per_class)
plip_terms1_pc = Path(snakemake.input.plip_terms1_per_class)
out_path = Path(snakemake.output.plot)
csv_out = Path(snakemake.output.csv_table)


metric = snakemake.params.metric


def read_pc(fp: Path) -> pd.DataFrame:
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
    return df


def load_baseline_logits(fp: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(fp)
    df = df[df["source_image"].str.contains("_patch.tiff")].copy()

    def parse_dataset_id(x: str) -> str:
        m = re.search(r"(reg\d+_[AB])", x)
        return m.group(1) if m else x

    df["dataset_id"] = df["source_image"].apply(parse_dataset_id)
    df["spot_id"] = df["spot_id"].astype(str)
    id_cols = {"source_image", "spot_id", "dataset_id"}
    class_cols = [c for c in df.columns if c not in id_cols]
    return df, class_cols


def load_logits_files(files: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    dfs = []
    for fp in files:
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        m = re.search(r"(reg\d+_[AB])", fp.stem)
        df["dataset_id"] = m.group(1)
        df["spot_id"] = df.index.map(
            lambda i: f"patch_{i}"
        )  # TODO change for cell-level support
        id_cols = {"spot_id", "dataset_id"}
        class_cols = [c for c in df.columns if c not in id_cols]

        if class_cols[0].isdigit():
            class_names = snakemake.params.class_names
            assert len(class_names) == len(class_cols)
            df.rename(
                columns={old: new for old, new in zip(class_cols, class_names)},
                inplace=True,
            )
            class_cols = class_names
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(), []
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    id_cols = {"source_image", "spot_id", "dataset_id"}
    class_cols = [c for c in df_all.columns if c not in id_cols]
    return df_all, class_cols


def compute_metrics_by_class(
    df: pd.DataFrame, class_cols: list[str], classes_ref: list[str]
) -> pd.DataFrame:
    scores = df[class_cols].astype(float).values
    scores = apply_score_norm(scores, mode=score_norm)
    y_true = [
        truth_map.get((row["dataset_id"], row.get("spot_id", str(i))))
        for i, row in df.iterrows()
    ]
    keep_idx = [
        i for i, t in enumerate(y_true) if (t is not None) and (t in classes_ref)
    ]
    if len(keep_idx) == 0:
        return pd.DataFrame(
            {"class_label": classes_ref, metric: [np.nan] * len(classes_ref)}
        )
    y_true_filt = [y_true[i] for i in keep_idx]
    scores_filt = scores[keep_idx, :]
    vals = []
    for j, cls in enumerate(class_cols):
        y_bin = np.array([1 if t == cls else 0 for t in y_true_filt])
        s = scores_filt[:, j]
        if metric == "f1":
            pred_idx = np.argmax(scores_filt, axis=1)
            y_pred_bin = (pred_idx == j).astype(int)
            tp = int(((y_bin == 1) & (y_pred_bin == 1)).sum())
            fp = int(((y_bin == 0) & (y_pred_bin == 1)).sum())
            fn = int(((y_bin == 1) & (y_pred_bin == 0)).sum())
            denom = 2 * tp + fp + fn
            val = (2 * tp / denom) if denom > 0 else np.nan
        elif metric in ("auroc", "rocauc"):
            try:
                val = roc_auc_score(y_bin, s)
            except Exception:
                val = np.nan
        else:
            val = np.nan
        vals.append((cls, val))
    val_map = {k: v for k, v in vals}
    out_vals = [val_map.get(cls, np.nan) for cls in classes_ref]
    return pd.DataFrame({"class_label": classes_ref, metric: out_vals})


bibridge_df = read_pc(bibridge_pc)
quilt_df = read_pc(quilt_pc)
conch_LLL_df = read_pc(conch_LLL_pc)
conch_LUL_df = read_pc(conch_LUL_pc)
conch_frozen_df = read_pc(conch_frozen_pc)
conch_terms1_df = read_pc(conch_terms1_pc)
plip_terms1_df = read_pc(plip_terms1_pc)


def make_plot_df(metric: str) -> pd.DataFrame:
    cols = ["class_label", metric]
    a = bibridge_df[cols].rename(columns={metric: "bibridge"})
    b = quilt_df[cols].rename(columns={metric: "bimodal_quilt1m"})
    cLLL = conch_LLL_df[cols].rename(columns={metric: "conch_LLL"})
    cLUL = conch_LUL_df[cols].rename(columns={metric: "conch_LUL"})
    cF = conch_frozen_df[cols].rename(columns={metric: "conch_frozen"})
    canimesh = plip_terms1_df[cols].rename(columns={metric: "plip_terms1"})
    panimesh = conch_terms1_df[cols].rename(columns={metric: "conch_terms1"})
    merged = a.merge(b, on="class_label", how="inner")
    merged = merged.merge(cLLL, on="class_label", how="left")
    merged = merged.merge(cLUL, on="class_label", how="left")
    merged = merged.merge(cF, on="class_label", how="left")
    merged = merged.merge(canimesh, on="class_label", how="left")
    merged = merged.merge(panimesh, on="class_label", how="left")

    # If baseline per-dataset scores are available, compute per-class metrics directly and append
    def per_class_from_scores(files: list[Path], label: str) -> pd.DataFrame:
        if not files:
            return pd.DataFrame(columns=["class_label", label])
        # Concatenate per-dataset and average per-class
        dfs = []
        for fp in files:
            df = pd.read_csv(fp)
            # Columns may be numeric indices; leave as-is. Align later via class_label mapping if needed.
            # Compute per-class metrics requires true labels; not available here, so we skip AUROC/F1 and only plot logits mean as proxy
            # Better: rely on aggregated per-class CSVs above for proper metrics.
        return pd.DataFrame(columns=["class_label", label])

    # Note: we keep baselines via aggregated per-class CSVs for proper metrics; the split scores are provided to enable future metrics_from_scores runs.
    long = merged.melt(id_vars=["class_label"], var_name="method", value_name="value")
    return long


method_colors = {
    "bibridge": "#f39c12",
    "bimodal_quilt1m": "#16a085",
    "conch_LLL": "#1f77b4",
    "conch_LUL": "#2ca02c",
    "conch_frozen": "#9467bd",
    "plip_terms1": "#7f7f7f",
    "conch_terms1": "#d62728",
}

fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(max(6, 0.25 * bibridge_df.shape[0]), 2.0),
)
dfm = make_plot_df(metric)
order = (
    dfm.groupby("class_label")["value"].mean().sort_values(ascending=False).index.tolist()
)
sns.barplot(
    data=dfm,
    x="class_label",
    y="value",
    hue="method",
    order=order,
    palette=method_colors,
    edgecolor="#333333",
    ax=ax,
)
ax.set_title(metric)
ax.tick_params(axis="x", rotation=60)
ax.set_xlabel("")
ax.set_ylabel("")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, title="Method", loc="upper right")
plt.tight_layout()
plt.savefig(out_path)
plt.close()


# Build CSV table for the selected metric: columns = classes (+ mean), rows = methods
cols = ["class_label", metric]
dfs = {
    "bibridge": bibridge_df[cols].rename(columns={metric: "bibridge"}),
    "bimodal_quilt1m": quilt_df[cols].rename(columns={metric: "bimodal_quilt1m"}),
    "conch_terms1": conch_terms1_df[cols].rename(columns={metric: "conch_terms1"}),
    "plip_terms1": plip_terms1_df[cols].rename(columns={metric: "plip_terms1"}),
}
merged = None
for label, df in dfs.items():
    merged = df if merged is None else merged.merge(df, on="class_label", how="inner")
exclude = {
    "Other cells",
    "Background",
    "A sample of Other cells",
    "A sample of Background cells",
}
merged = merged[~merged["class_label"].isin(exclude)].copy()
merged = merged.set_index("class_label")
# Append mean over classes
merged.loc["mean"] = merged.mean(numeric_only=True)

merged.to_csv(csv_out, index=True)
