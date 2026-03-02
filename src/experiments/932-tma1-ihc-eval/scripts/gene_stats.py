#!/usr/bin/env python
"""Inspect gene variance/mean/sparsity distribution in ground truth."""
import anndata
import numpy as np
import pandas as pd

PRED_DIR = "/home/users/moritzs/cellwhisperer_private/results/experiments/932-tma1-ihc-eval/predictions"
OUT_DIR = "/scratch/users/moritzs"

for tma in ["TMA2", "TMA3"]:
    a = anndata.read_h5ad(f"{PRED_DIR}/{tma}_predictions.h5ad")
    gt = np.asarray(a.layers["ground_truth"])
    gene_mean = gt.mean(axis=0)
    gene_var = gt.var(axis=0)
    gene_nonzero_frac = (gt > 0).mean(axis=0)

    df = pd.DataFrame({"gene": a.var_names, "mean": gene_mean, "var": gene_var, "nonzero_frac": gene_nonzero_frac})
    df.to_csv(f"{OUT_DIR}/{tma}_gene_stats.csv", index=False)

    print(f"\n{tma}: {gt.shape[0]} cells, {gt.shape[1]} genes")
    for q in [50, 75, 90, 95, 99]:
        print(f"  var p{q}: {np.percentile(gene_var, q):.6f}  mean p{q}: {np.percentile(gene_mean, q):.6f}  nonzero_frac p{q}: {np.percentile(gene_nonzero_frac, q):.4f}")
    print(f"  genes with var>0.01: {(gene_var > 0.01).sum()}, var>0.1: {(gene_var > 0.1).sum()}, var>1: {(gene_var > 1.0).sum()}")
    print(f"  genes with mean>0.01: {(gene_mean > 0.01).sum()}, mean>0.1: {(gene_mean > 0.1).sum()}")
    print(f"  genes with nonzero_frac>0.01: {(gene_nonzero_frac > 0.01).sum()}, >0.05: {(gene_nonzero_frac > 0.05).sum()}, >0.1: {(gene_nonzero_frac > 0.1).sum()}")
    # also: top 20 most variable genes
    top20 = df.nlargest(20, "var")
    print(f"  Top 20 most variable genes:")
    for _, row in top20.iterrows():
        print(f"    {row['gene']:12s}  var={row['var']:.4f}  mean={row['mean']:.4f}  nonzero={row['nonzero_frac']:.4f}")
