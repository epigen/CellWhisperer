#!/usr/bin/env python
"""
Correlate core-level base model metrics (retrieval, loss) with decoder performance.

Merges per-core CSVs and computes pairwise correlations. Produces:
  - Combined CSV with all core-level metrics
  - Correlation matrix CSV
  - Correlation heatmap
  - Scatter plots for key relationships

Usage:
    python correlate_core_metrics.py \
        --retrieval_csvs TMA2_core_retrieval.csv \
        --loss_csvs TMA2_core_loss.csv \
        --decoder_csvs TMA2_core_decoder.csv \
        --output_combined combined.csv \
        --output_corr_matrix correlation_matrix.csv \
        --output_plots plots_dir/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


# Key metric pairs to highlight in scatter plots
KEY_PAIRS = [
    ("i2t_recall_at_1_macroAvg", "median_gene_pearson"),
    ("i2t_recall_at_5_macroAvg", "median_gene_pearson"),
    ("t2i_recall_at_1_macroAvg", "median_gene_pearson"),
    ("mean_loss", "median_gene_pearson"),
    ("mean_loss", "mae"),
    ("n_cells", "median_gene_pearson"),
]


def load_and_merge(retrieval_csvs, loss_csvs, decoder_csvs):
    """Load all CSVs and merge on (tma, core_id)."""
    dfs_ret = [pd.read_csv(p) for p in retrieval_csvs]
    dfs_loss = [pd.read_csv(p) for p in loss_csvs]
    dfs_dec = [pd.read_csv(p) for p in decoder_csvs]

    df_ret = pd.concat(dfs_ret, ignore_index=True)
    df_loss = pd.concat(dfs_loss, ignore_index=True)
    df_dec = pd.concat(dfs_dec, ignore_index=True)

    # Merge: retrieval + loss share (tma, core_id, n_cells)
    df = df_ret.merge(
        df_loss.drop(columns=["n_cells"], errors="ignore"),
        on=["tma", "core_id"],
        how="inner",
    )
    df = df.merge(
        df_dec.drop(columns=["n_cells"], errors="ignore"),
        on=["tma", "core_id"],
        how="inner",
    )
    return df


def compute_correlation_matrix(df, metric_cols):
    """Compute pairwise Pearson and Spearman correlations."""
    n = len(metric_cols)
    pearson_r = np.full((n, n), np.nan)
    pearson_p = np.full((n, n), np.nan)
    spearman_r = np.full((n, n), np.nan)
    spearman_p = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):
            x = df[metric_cols[i]].values
            y = df[metric_cols[j]].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() >= 3:
                r, p = pearsonr(x[valid], y[valid])
                pearson_r[i, j] = r
                pearson_p[i, j] = p
                r, p = spearmanr(x[valid], y[valid])
                spearman_r[i, j] = r
                spearman_p[i, j] = p

    df_pearson = pd.DataFrame(pearson_r, index=metric_cols, columns=metric_cols)
    df_pearson_p = pd.DataFrame(pearson_p, index=metric_cols, columns=metric_cols)
    df_spearman = pd.DataFrame(spearman_r, index=metric_cols, columns=metric_cols)
    df_spearman_p = pd.DataFrame(spearman_p, index=metric_cols, columns=metric_cols)

    return df_pearson, df_pearson_p, df_spearman, df_spearman_p


def plot_heatmap(corr_matrix, p_matrix, output_path, title="Pearson Correlation"):
    """Plot correlation heatmap with significance markers."""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Mask for significance
    sig_mask = p_matrix < 0.05
    annot = corr_matrix.round(2).astype(str)
    annot[sig_mask] = annot[sig_mask] + "*"
    annot[p_matrix < 0.01] = corr_matrix[p_matrix < 0.01].round(2).astype(str) + "**"

    sns.heatmap(
        corr_matrix,
        annot=annot,
        fmt="s",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 7},
    )
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap: {output_path}")


def plot_scatter(df, x_col, y_col, output_path):
    """Scatter plot with regression line, Pearson + Spearman annotations."""
    x = df[x_col].values
    y = df[y_col].values
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    if len(x) < 3:
        return

    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Color by TMA if available
    if "tma" in df.columns:
        tmas = df.loc[valid, "tma"].values if valid.sum() == len(df) else df["tma"].values
        for tma_val in sorted(set(tmas)):
            mask = tmas == tma_val
            ax.scatter(x[mask], y[mask], alpha=0.7, label=tma_val, s=50, edgecolor="k", linewidth=0.5)
        ax.legend()
    else:
        ax.scatter(x, y, alpha=0.7, s=50, edgecolor="k", linewidth=0.5)

    # Regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=1.5)

    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)
    ax.set_title(
        f"Pearson r={r_p:.3f} (p={p_p:.2e})\nSpearman ρ={r_s:.3f} (p={p_s:.2e})",
        fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_csvs", type=Path, nargs="+", required=True)
    parser.add_argument("--loss_csvs", type=Path, nargs="+", required=True)
    parser.add_argument("--decoder_csvs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_combined", type=Path, required=True)
    parser.add_argument("--output_corr_matrix", type=Path, required=True)
    parser.add_argument("--output_plots", type=Path, required=True)
    args = parser.parse_args()

    args.output_combined.parent.mkdir(parents=True, exist_ok=True)
    args.output_corr_matrix.parent.mkdir(parents=True, exist_ok=True)
    args.output_plots.mkdir(parents=True, exist_ok=True)

    # Load and merge
    df = load_and_merge(args.retrieval_csvs, args.loss_csvs, args.decoder_csvs)
    print(f"Combined data: {len(df)} cores")
    df.to_csv(args.output_combined, index=False)
    print(f"Saved combined CSV: {args.output_combined}")

    # Select numeric metric columns (exclude tma, core_id)
    exclude = {"tma", "core_id"}
    metric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    # Compute correlations
    print(f"Computing correlations for {len(metric_cols)} metrics...")
    df_pearson, df_pearson_p, df_spearman, df_spearman_p = compute_correlation_matrix(df, metric_cols)

    # Save correlation matrix
    df_pearson.to_csv(args.output_corr_matrix)
    df_spearman.to_csv(args.output_corr_matrix.with_name("spearman_" + args.output_corr_matrix.name))
    print(f"Saved correlation matrices")

    # Plot heatmap (select subset of most relevant metrics)
    relevant_metrics = [
        c for c in metric_cols
        if any(kw in c for kw in [
            "recall_at_1_macro", "recall_at_5_macro", "recall_at_10_macro",
            "accuracy_macro", "f1_macro", "rocauc_macro",
            "mean_loss", "median_loss",
            "mean_gene_pearson", "median_gene_pearson", "mae", "mse",
            "n_cells", "n_genes",
        ])
    ]
    if relevant_metrics:
        sub_pearson = df_pearson.loc[relevant_metrics, relevant_metrics]
        sub_pearson_p = df_pearson_p.loc[relevant_metrics, relevant_metrics]
        plot_heatmap(sub_pearson, sub_pearson_p, args.output_plots / "correlation_heatmap.png")

    # Scatter plots for key pairs
    scatter_dir = args.output_plots / "scatter_plots"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    for x_col, y_col in KEY_PAIRS:
        if x_col in df.columns and y_col in df.columns:
            fname = f"{x_col}_vs_{y_col}.png"
            plot_scatter(df, x_col, y_col, scatter_dir / fname)
            print(f"  Scatter: {x_col} vs {y_col}")

    # Print summary of strongest correlations with decoder performance
    decoder_metrics = ["median_gene_pearson", "mean_gene_pearson", "mae", "mse"]
    base_metrics = [c for c in metric_cols if c not in decoder_metrics and c not in ["n_cells", "n_genes_evaluated"]]

    print("\n=== Strongest correlations with decoder performance ===")
    for target in decoder_metrics:
        if target not in df.columns:
            continue
        print(f"\n  Target: {target}")
        corrs = []
        for pred in base_metrics:
            if pred not in df.columns:
                continue
            x = df[pred].values
            y = df[target].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() >= 3:
                r, p = pearsonr(x[valid], y[valid])
                corrs.append((pred, r, p))
        corrs.sort(key=lambda t: abs(t[1]), reverse=True)
        for pred, r, p in corrs[:10]:
            sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {pred:40s}  r={r:+.3f}  p={p:.2e} {sig}")


if __name__ == "__main__":
    main()
