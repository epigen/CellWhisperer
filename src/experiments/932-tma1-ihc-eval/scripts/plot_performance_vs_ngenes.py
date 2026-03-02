#!/usr/bin/env python
"""
Plot performance vs number of genes considered.
Creates line plots with x-axis = n_genes and y-axis = metric values.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Input/output paths
csv_path = Path("/home/moritz/code/cellwhisperer/results/experiments/932-tma1-ihc-eval/metrics/metrics_with_baseline.csv")
out_dir = Path("/home/moritz/code/cellwhisperer/results/experiments/932-tma1-ihc-eval/plots/baseline_comparison")

df = pd.read_csv(csv_path)

# Filter to top-N subsets only (not "all" or "nzf" filters)
df_topn = df[df["subset"].str.match(r"^top\d+$", na=False)].copy()

# Sort by n_genes for proper line plotting
df_topn = df_topn.sort_values("n_genes")

# Metrics to plot
metrics = ["mae", "mse", "mean_gene_pearson", "median_gene_pearson"]
metric_labels = {
    "mae": "MAE",
    "mse": "MSE",
    "mean_gene_pearson": "Mean per-gene Pearson r",
    "median_gene_pearson": "Median per-gene Pearson r"
}

for tma in ["TMA2", "TMA3"]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, metric in zip(axes, metrics):
        df_tma = df_topn[df_topn["tma"] == tma]
        
        for method in ["model", "scrambled"]:
            df_method = df_tma[df_tma["method"] == method]
            ax.plot(df_method["n_genes"], df_method[metric], 
                   marker="o", label=method.capitalize(), linewidth=2)
        
        ax.set_xlabel("Number of genes (top-N HVGs)", fontsize=11)
        ax.set_ylabel(metric_labels[metric], fontsize=11)
        ax.set_xscale("log")
        ax.set_xticks([50, 100, 500, 1000])
        ax.set_xticklabels(["50", "100", "500", "1000"])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{metric_labels[metric]}", fontsize=12, fontweight="bold")
    
    fig.suptitle(f"{tma}: Performance vs Number of Genes", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / f"{tma}_performance_vs_ngenes.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / f'{tma}_performance_vs_ngenes.png'}")

print("Done.")
