#!/usr/bin/env python
"""
Evaluate gene expression predictions against ground truth with scrambled baselines.

For each TMA, computes MAE, MSE, and per-gene Pearson correlation for:
  - Model predictions vs ground truth
  - Scrambled baseline (ground truth rows shuffled globally per gene)
  - Within-core scrambled baseline (ground truth shuffled within each core per gene)

Reports metrics for multiple gene subsets (all genes, top-N by GT variance,
and genes with nonzero fraction above a threshold).

Usage:
    python evaluate_with_baseline.py \
        --predictions TMA2_predictions.h5ad TMA3_predictions.h5ad \
        --output_csv metrics_with_baseline.csv \
        --output_plots plots_dir/
"""

import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Gene subsets to evaluate: (label, N_top_genes_by_gt_variance)
# None means all genes
TOP_N_SUBSETS = [None, 50, 100, 500, 1000]
# Additional filter: minimum fraction of nonzero cells in ground truth
NONZERO_FRAC_THRESHOLDS = [0.0, 0.05, 0.10]


def per_gene_pearson(pred, gt, gene_names):
    """Return dict of gene_name -> Pearson r for valid genes."""
    out = {}
    for j in range(pred.shape[1]):
        if np.std(pred[:, j]) > 0 and np.std(gt[:, j]) > 0:
            r, _ = pearsonr(pred[:, j], gt[:, j])
            if not np.isnan(r):
                out[gene_names[j]] = r
    return out


def global_metrics(pred, gt):
    """MAE and MSE over all entries."""
    return {
        "mae": mean_absolute_error(gt.flatten(), pred.flatten()),
        "mse": mean_squared_error(gt.flatten(), pred.flatten()),
    }


def scramble_ground_truth(gt, rng):
    """Shuffle ground truth independently per gene (column) across all cells."""
    shuffled = gt.copy()
    for j in range(shuffled.shape[1]):
        rng.shuffle(shuffled[:, j])
    return shuffled


def scramble_within_cores(gt, core_ids, rng):
    """Shuffle ground truth within each core independently per gene."""
    shuffled = gt.copy()
    unique_cores = np.unique(core_ids)
    for core in unique_cores:
        mask = core_ids == core
        for j in range(shuffled.shape[1]):
            # Shuffle only within this core for this gene
            core_vals = shuffled[mask, j].copy()
            rng.shuffle(core_vals)
            shuffled[mask, j] = core_vals
    return shuffled


def select_gene_indices(gt, gene_names, top_n=None, min_nonzero_frac=0.0):
    """Return boolean mask of genes passing the filter."""
    gt_var = gt.var(axis=0)
    nonzero_frac = (gt > 0).mean(axis=0)
    mask = nonzero_frac >= min_nonzero_frac
    if top_n is not None:
        # Among passing genes, keep only top_n by variance
        var_masked = np.where(mask, gt_var, -np.inf)
        topk_idx = np.argsort(var_masked)[-top_n:]
        new_mask = np.zeros_like(mask)
        new_mask[topk_idx] = True
        mask = mask & new_mask
    return mask


def summarize_pearsons(gene_r):
    vals = list(gene_r.values())
    return {
        "mean_gene_pearson": np.mean(vals) if vals else np.nan,
        "median_gene_pearson": np.median(vals) if vals else np.nan,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_plots", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_plots.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_gene_rows = []

    for pred_path in args.predictions:
        tma = pred_path.stem.replace("_predictions", "")
        print(f"Loading {pred_path.name} ...")
        adata = anndata.read_h5ad(pred_path)
        predictions = np.asarray(adata.X)
        ground_truth = np.asarray(adata.layers["ground_truth"])
        gene_names = np.array(adata.var_names.tolist())
        core_ids = adata.obs["core_id"].values

        gt_shuffled = scramble_ground_truth(ground_truth, rng)
        gt_shuffled_within_core = scramble_within_cores(ground_truth, core_ids, rng)

        # Per-gene Pearson on ALL genes (for gene-level CSV + plots)
        model_gene_r = per_gene_pearson(predictions, ground_truth, gene_names)
        baseline_gene_r = per_gene_pearson(predictions, gt_shuffled, gene_names)
        baseline_wc_gene_r = per_gene_pearson(predictions, gt_shuffled_within_core, gene_names)
        common = sorted(set(model_gene_r) & set(baseline_gene_r) & set(baseline_wc_gene_r))
        for g in common:
            all_gene_rows.append({
                "tma": tma, "gene": g, 
                "model_r": model_gene_r[g], 
                "scrambled_r": baseline_gene_r[g],
                "scrambled_within_core_r": baseline_wc_gene_r[g]
            })

        # Evaluate each (top_n, nonzero_frac) subset
        for min_nzf in NONZERO_FRAC_THRESHOLDS:
            for top_n in TOP_N_SUBSETS:
                mask = select_gene_indices(ground_truth, gene_names, top_n=top_n, min_nonzero_frac=min_nzf)
                n_genes = mask.sum()
                if n_genes == 0:
                    continue
                label = f"top{top_n}" if top_n else "all"
                if min_nzf > 0:
                    label += f"_nzf>{min_nzf}"

                pred_sub = predictions[:, mask]
                gt_sub = ground_truth[:, mask]
                gt_shuf_sub = gt_shuffled[:, mask]
                gt_shuf_wc_sub = gt_shuffled_within_core[:, mask]
                gn_sub = gene_names[mask]

                for method, p, g in [("model", pred_sub, gt_sub), 
                                      ("scrambled", pred_sub, gt_shuf_sub),
                                      ("scrambled_within_core", pred_sub, gt_shuf_wc_sub)]:
                    gm = global_metrics(p, g)
                    gr = per_gene_pearson(p, g, gn_sub)
                    ps = summarize_pearsons(gr)
                    summary_rows.append({
                        "tma": tma, "method": method, "subset": label,
                        "n_genes": int(n_genes),
                        **gm, **ps,
                    })

                print(f"  {tma} | {label:25s} | {n_genes:5d} genes | model={summary_rows[-3]['median_gene_pearson']:.4f} | scrambled={summary_rows[-2]['median_gene_pearson']:.4f} | within_core={summary_rows[-1]['median_gene_pearson']:.4f}")

        # --- Plot: histogram for top-500 subset ---
        mask500 = select_gene_indices(ground_truth, gene_names, top_n=500)
        gn500 = gene_names[mask500]
        mr500 = per_gene_pearson(predictions[:, mask500], ground_truth[:, mask500], gn500)
        br500 = per_gene_pearson(predictions[:, mask500], gt_shuffled[:, mask500], gn500)
        bwc500 = per_gene_pearson(predictions[:, mask500], gt_shuffled_within_core[:, mask500], gn500)
        common500 = sorted(set(mr500) & set(br500) & set(bwc500))
        m_vals = np.array([mr500[g] for g in common500])
        b_vals = np.array([br500[g] for g in common500])
        bwc_vals = np.array([bwc500[g] for g in common500])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(m_vals, bins=40, alpha=0.5, label=f"Model (median={np.median(m_vals):.3f})")
        ax.hist(b_vals, bins=40, alpha=0.5, label=f"Scrambled (median={np.median(b_vals):.3f})")
        ax.hist(bwc_vals, bins=40, alpha=0.5, label=f"Within-core (median={np.median(bwc_vals):.3f})")
        ax.set_xlabel("Per-gene Pearson r")
        ax.set_ylabel("Number of genes")
        ax.set_title(f"{tma}: Per-gene Pearson (top 500 HVGs)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_plots / f"{tma}_gene_pearson_top500.png", dpi=150)
        plt.close(fig)

    # Save
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_csv, index=False)
    print(f"\nSummary saved to {args.output_csv}")
    # Print a compact view
    for tma in summary_df["tma"].unique():
        print(f"\n=== {tma} ===")
        sub = summary_df[summary_df["tma"] == tma].pivot(index="subset", columns="method", values="median_gene_pearson")
        sub = sub.reindex(columns=["model", "scrambled", "scrambled_within_core"])
        print(sub.to_string())

    gene_df = pd.DataFrame(all_gene_rows)
    gene_df.to_csv(args.output_csv.with_name("gene_level_metrics.csv"), index=False)


if __name__ == "__main__":
    main()
