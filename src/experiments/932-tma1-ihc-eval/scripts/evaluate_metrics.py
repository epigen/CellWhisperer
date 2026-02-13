#!/usr/bin/env python
"""
Evaluate gene expression predictions against ground truth.

Computes evaluation metrics (MSE, MAE, correlation) comparing predicted
gene expression to ground truth from the original dataset.
"""

import argparse
import logging
from pathlib import Path
import sys

import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cellwhisperer.config import get_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_metrics(predictions, ground_truth, gene_names):
    """Compute evaluation metrics."""
    metrics = {}
    
    # Overall metrics
    metrics["mse"] = mean_squared_error(ground_truth.flatten(), predictions.flatten())
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = mean_absolute_error(ground_truth.flatten(), predictions.flatten())
    
    # Per-sample correlations
    sample_correlations = []
    for i in range(predictions.shape[0]):
        if np.std(predictions[i]) > 0 and np.std(ground_truth[i]) > 0:
            corr, _ = pearsonr(predictions[i], ground_truth[i])
            if not np.isnan(corr):
                sample_correlations.append(corr)
    
    metrics["mean_sample_pearson"] = np.mean(sample_correlations) if sample_correlations else 0.0
    metrics["median_sample_pearson"] = np.median(sample_correlations) if sample_correlations else 0.0
    metrics["std_sample_pearson"] = np.std(sample_correlations) if sample_correlations else 0.0
    
    # Per-gene correlations
    gene_correlations = []
    gene_correlation_dict = {}
    for j in range(predictions.shape[1]):
        if np.std(predictions[:, j]) > 0 and np.std(ground_truth[:, j]) > 0:
            corr, _ = pearsonr(predictions[:, j], ground_truth[:, j])
            if not np.isnan(corr):
                gene_correlations.append(corr)
                gene_correlation_dict[gene_names[j]] = corr
    
    metrics["mean_gene_pearson"] = np.mean(gene_correlations) if gene_correlations else 0.0
    metrics["median_gene_pearson"] = np.median(gene_correlations) if gene_correlations else 0.0
    metrics["std_gene_pearson"] = np.std(gene_correlations) if gene_correlations else 0.0
    
    return metrics, sample_correlations, gene_correlation_dict


def create_plots(predictions, ground_truth, gene_names, sample_correlations, 
                gene_correlation_dict, output_dir, dataset_name):
    """Create evaluation plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Scatter plot: predicted vs ground truth (all genes, all samples)
    plt.figure(figsize=(8, 8))
    # Subsample for visualization if too many points
    n_points = predictions.size
    if n_points > 100000:
        indices = np.random.choice(n_points, 100000, replace=False)
        pred_flat = predictions.flatten()[indices]
        gt_flat = ground_truth.flatten()[indices]
    else:
        pred_flat = predictions.flatten()
        gt_flat = ground_truth.flatten()
    
    plt.scatter(gt_flat, pred_flat, alpha=0.1, s=1)
    plt.plot([gt_flat.min(), gt_flat.max()], [gt_flat.min(), gt_flat.max()], 
             'r--', lw=2, label='Perfect prediction')
    plt.xlabel('Ground Truth Expression (log)')
    plt.ylabel('Predicted Expression (log)')
    plt.title(f'Predicted vs Ground Truth\n{dataset_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_pred_vs_gt.png", dpi=150)
    plt.close()
    
    # 2. Distribution of per-sample correlations
    if sample_correlations:
        plt.figure(figsize=(10, 6))
        plt.hist(sample_correlations, bins=50, edgecolor='black')
        plt.axvline(np.mean(sample_correlations), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(sample_correlations):.3f}')
        plt.axvline(np.median(sample_correlations), color='g', linestyle='--', 
                   label=f'Median: {np.median(sample_correlations):.3f}')
        plt.xlabel('Pearson Correlation')
        plt.ylabel('Number of Samples')
        plt.title(f'Distribution of Per-Sample Correlations\n{dataset_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "hist_sample_correlations.png", dpi=150)
        plt.close()
    
    # 3. Distribution of per-gene correlations
    if gene_correlation_dict:
        gene_corrs = list(gene_correlation_dict.values())
        plt.figure(figsize=(10, 6))
        plt.hist(gene_corrs, bins=50, edgecolor='black')
        plt.axvline(np.mean(gene_corrs), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(gene_corrs):.3f}')
        plt.axvline(np.median(gene_corrs), color='g', linestyle='--', 
                   label=f'Median: {np.median(gene_corrs):.3f}')
        plt.xlabel('Pearson Correlation')
        plt.ylabel('Number of Genes')
        plt.title(f'Distribution of Per-Gene Correlations\n{dataset_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "hist_gene_correlations.png", dpi=150)
        plt.close()
    
    # 4. Top and bottom genes by correlation
    if gene_correlation_dict:
        gene_corr_df = pd.DataFrame({
            'gene': list(gene_correlation_dict.keys()),
            'correlation': list(gene_correlation_dict.values())
        }).sort_values('correlation', ascending=False)
        
        # Plot top 20 and bottom 20
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top 20
        top_20 = gene_corr_df.head(20)
        ax1.barh(range(len(top_20)), top_20['correlation'])
        ax1.set_yticks(range(len(top_20)))
        ax1.set_yticklabels(top_20['gene'])
        ax1.set_xlabel('Pearson Correlation')
        ax1.set_title('Top 20 Genes by Correlation')
        ax1.invert_yaxis()
        
        # Bottom 20
        bottom_20 = gene_corr_df.tail(20)
        ax2.barh(range(len(bottom_20)), bottom_20['correlation'])
        ax2.set_yticks(range(len(bottom_20)))
        ax2.set_yticklabels(bottom_20['gene'])
        ax2.set_xlabel('Pearson Correlation')
        ax2.set_title('Bottom 20 Genes by Correlation')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_dir / "top_bottom_genes.png", dpi=150)
        plt.close()
        
        # Save gene correlations to CSV
        gene_corr_df.to_csv(output_dir / "gene_correlations.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate gene expression predictions")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Path to predictions h5ad file")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name for loading ground truth")
    parser.add_argument("--output_metrics", type=Path, required=True,
                        help="Output path for metrics CSV")
    parser.add_argument("--output_plots", type=Path, required=True,
                        help="Output directory for plots")
    
    args = parser.parse_args()
    
    logger.info(f"Loading predictions from: {args.predictions}")
    
    # Load predictions
    pred_adata = anndata.read_h5ad(args.predictions)
    predictions = pred_adata.X
    
    # Get ground truth from the predictions file (stored in .layers during prediction)
    if "ground_truth" in pred_adata.layers:
        ground_truth = pred_adata.layers["ground_truth"]
        logger.info("Using ground truth from predictions file")
    else:
        logger.error("No ground truth found in predictions file")
        raise ValueError("Ground truth not found in predictions h5ad file")
    
    gene_names = pred_adata.var_names.tolist()
    
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Ground truth shape: {ground_truth.shape}")
    logger.info(f"Number of genes: {len(gene_names)}")
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics, sample_correlations, gene_correlation_dict = compute_metrics(
        predictions, ground_truth, gene_names
    )
    
    # Log metrics
    logger.info("=" * 60)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 60)
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.6f}")
    logger.info("=" * 60)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df["dataset"] = args.dataset_name
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.output_metrics, index=False)
    logger.info(f"Metrics saved to: {args.output_metrics}")
    
    # Create plots
    logger.info("Creating plots...")
    create_plots(
        predictions, ground_truth, gene_names,
        sample_correlations, gene_correlation_dict,
        args.output_plots, args.dataset_name
    )
    logger.info(f"Plots saved to: {args.output_plots}")
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
