#!/usr/bin/env python3
"""
Build AUROC per-class comparison table for baseline variants:
- conch_terms1 vs conch_terms2
- plip_terms1 vs plip_terms2

Inputs (Snakemake):
- snakemake.input.conch_terms1_per_class
- snakemake.input.conch_terms2_per_class
- snakemake.input.plip_terms1_per_class
- snakemake.input.plip_terms2_per_class

Output:
- snakemake.output.csv_table

Behavior mirrors the CSV table creation in plot_pathocell_baselines_vs_trimodal.py,
but restricted to AUROC and these four baselines.
"""

from pathlib import Path
import pandas as pd


def read_pc(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
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


conch_terms1 = read_pc(Path(snakemake.input.conch_terms1_per_class))
conch_terms2 = read_pc(Path(snakemake.input.conch_terms2_per_class))
plip_terms1 = read_pc(Path(snakemake.input.plip_terms1_per_class))
plip_terms2 = read_pc(Path(snakemake.input.plip_terms2_per_class))

metric = "rocauc"
cols = ["class_label", metric]

dfs = {
    "conch_terms1": conch_terms1[cols].rename(columns={metric: "conch_terms1"}),
    "conch_terms2": conch_terms2[cols].rename(columns={metric: "conch_terms2"}),
    "plip_terms1": plip_terms1[cols].rename(columns={metric: "plip_terms1"}),
    "plip_terms2": plip_terms2[cols].rename(columns={metric: "plip_terms2"}),
}

merged = None
for _, df in dfs.items():
    merged = df if merged is None else merged.merge(df, on="class_label", how="inner")

# Drop background/other classes for readability, consistent with baseline script
exclude = {
    "Other cells",
    "Background",
    "A sample of Other cells",
    "A sample of Background cells",
}
merged = merged[~merged["class_label"].isin(exclude)].copy()

merged = merged.set_index("class_label")
merged.loc["mean"] = merged.mean(numeric_only=True)


def mark_best(series: pd.Series) -> pd.Series:
    vals = series.astype(float).copy()
    order = vals.sort_values(ascending=False).index.tolist()
    out = series.copy().astype(str)
    if len(order) > 0:
        b = order[0]
        out[b] = f"**{vals[b]:.3f}**"
    if len(order) > 1:
        s = order[1]
        out[s] = f"__{vals[s]:.3f}__"
    for idx in order[2:]:
        out[idx] = f"{vals[idx]:.3f}"
    return out


formatted = merged.apply(mark_best, axis=0)

out_csv = Path(snakemake.output.csv_table)
out_csv.parent.mkdir(parents=True, exist_ok=True)
formatted.to_csv(out_csv, index=True)
