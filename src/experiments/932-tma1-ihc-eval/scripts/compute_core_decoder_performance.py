#!/usr/bin/env python
"""
Compute per-core gene expression decoder performance metrics.

For each core in a TMA prediction file, computes:
  - Per-gene Pearson correlation (model vs ground truth)
  - Mean/median Pearson across genes
  - MAE, MSE

Usage:
    python compute_core_decoder_performance.py \
        --predictions TMA2_predictions.h5ad \
        --top_n 500 \
        --output core_decoder.csv
"""

import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def per_gene_pearson(pred, gt):
    """Return array of per-gene Pearson r (NaN for invalid genes)."""
    n_genes = pred.shape[1]
    rs = np.full(n_genes, np.nan)
    for j in range(n_genes):
        if np.std(pred[:, j]) > 0 and np.std(gt[:, j]) > 0:
            r, _ = pearsonr(pred[:, j], gt[:, j])
            rs[j] = r
    return rs


def select_top_hvg_indices(gt, top_n):
    """Return indices of top_n genes by variance across all cells."""
    var = gt.var(axis=0)
    return np.argsort(var)[-top_n:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--top_n", type=int, default=500)
    parser.add_argument("--min_cells", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.predictions} ...")
    adata = anndata.read_h5ad(args.predictions)
    tma = args.predictions.stem.replace("_predictions", "")

    predictions = np.asarray(adata.X)
    ground_truth = np.asarray(adata.layers["ground_truth"])
    gene_names = np.array(adata.var_names.tolist())
    core_ids = adata.obs["core_id"].values

    # Select top-N HVGs (based on global GT variance)
    hvg_idx = select_top_hvg_indices(ground_truth, args.top_n)
    predictions = predictions[:, hvg_idx]
    ground_truth = ground_truth[:, hvg_idx]
    gene_names = gene_names[hvg_idx]
    print(f"Selected top {args.top_n} HVGs ({len(gene_names)} genes)")

    unique_cores = np.unique(core_ids)
    rows = []

    for core in unique_cores:
        mask = core_ids == core
        n_cells = mask.sum()
        if n_cells < args.min_cells:
            print(f"  Skipping core {core} ({n_cells} cells < {args.min_cells})")
            continue

        pred_core = predictions[mask]
        gt_core = ground_truth[mask]

        # Per-gene Pearson
        gene_rs = per_gene_pearson(pred_core, gt_core)
        valid_rs = gene_rs[~np.isnan(gene_rs)]

        # MAE, MSE
        mae = np.mean(np.abs(pred_core - gt_core))
        mse = np.mean((pred_core - gt_core) ** 2)

        rows.append({
            "tma": tma,
            "core_id": core,
            "n_cells": n_cells,
            "n_genes_evaluated": len(valid_rs),
            "mean_gene_pearson": np.mean(valid_rs) if len(valid_rs) > 0 else np.nan,
            "median_gene_pearson": np.median(valid_rs) if len(valid_rs) > 0 else np.nan,
            "mae": mae,
            "mse": mse,
        })
        print(f"  Core {core}: {n_cells} cells, median_r={rows[-1]['median_gene_pearson']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} cores to {args.output}")


if __name__ == "__main__":
    main()
