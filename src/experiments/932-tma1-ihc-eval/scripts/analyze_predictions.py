"""
Analyze gene expression predictions from TMA3 and correlate with IHC ground truth.

This script:
1. Loads predicted gene expression from h5ad file
2. Aggregates predictions by core (mean and 90th percentile)
3. Correlates predicted PAX5 and CD19 expression with histopathologist IHC scores
4. Creates scatter plots showing correlations
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

def load_predictions(prediction_file):
    """Load predictions from h5ad file."""
    print(f"Loading predictions from {prediction_file}...")
    adata = anndata.read_h5ad(prediction_file)
    print(f"Loaded {adata.shape[0]} cells x {adata.shape[1]} genes")
    return adata

def load_ihc_ground_truth(ihc_file):
    """Load IHC ground truth scores from Excel file."""
    print(f"Loading IHC ground truth from {ihc_file}...")
    df = pd.read_excel(ihc_file, header=1)
    
    # Keep only relevant columns
    df = df[['Blocks', 'core_id', 'CD19', 'CD20 H-score', 'PAX5 H-score']].copy()
    df = df.rename(columns={
        'CD20 H-score': 'CD20_Hscore',
        'PAX5 H-score': 'PAX5_Hscore',
        'CD19': 'CD19_Hscore'
    })
    
    print(f"Loaded {len(df)} cores")
    print(f"  CD19 scores: {df['CD19_Hscore'].notna().sum()} non-null")
    print(f"  CD20 scores: {df['CD20_Hscore'].notna().sum()} non-null")
    print(f"  PAX5 scores: {df['PAX5_Hscore'].notna().sum()} non-null")
    
    return df

def filter_variable_genes(adata, std_threshold=0.1):
    """Filter for genes with sufficient variability (non-boring genes)."""
    # Calculate std on predicted expression (which is in .X)
    gene_std = np.std(adata.X, axis=0)
    if hasattr(gene_std, 'A1'):  # If sparse matrix
        gene_std = gene_std.A1
    
    variable_mask = gene_std > std_threshold
    n_variable = variable_mask.sum()
    
    print(f"Filtering genes with std > {std_threshold}")
    print(f"  Kept {n_variable}/{len(variable_mask)} genes ({100*n_variable/len(variable_mask):.1f}%)")
    print(f"  Std range: {gene_std.min():.3f} - {gene_std.max():.3f}")
    
    if n_variable == 0:
        print(f"  WARNING: No genes pass threshold, using all genes instead")
        return adata
    
    return adata[:, variable_mask].copy()

def aggregate_by_core(adata, aggregation='mean'):
    """
    Aggregate expression by core.
    
    Args:
        adata: AnnData object with predictions
        aggregation: 'mean' or 'p90' (90th percentile)
    
    Returns:
        DataFrame with aggregated expression per core
    """
    # Extract core information from obs
    # Prefer core_id over fov for proper core identification
    if 'core_id' in adata.obs.columns:
        core_column = 'core_id'
    elif 'core' in adata.obs.columns:
        core_column = 'core'
    elif 'fov' in adata.obs.columns:
        core_column = 'fov'
    else:
        raise ValueError(f"Could not find core identifier in obs. Available columns: {adata.obs.columns.tolist()}")
    
    print(f"Aggregating by '{core_column}' using {aggregation}")
    
    # Convert to dense if sparse
    if hasattr(adata.X, 'toarray'):
        expr_matrix = adata.X.toarray()
    else:
        expr_matrix = adata.X
    
    # Create DataFrame
    expr_df = pd.DataFrame(
        expr_matrix,
        index=adata.obs.index,
        columns=adata.var_names
    )
    expr_df[core_column] = adata.obs[core_column].values
    
    # Aggregate
    if aggregation == 'mean':
        agg_df = expr_df.groupby(core_column).mean()
    elif aggregation == 'p90':
        agg_df = expr_df.groupby(core_column).quantile(0.9)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    print(f"  Aggregated to {len(agg_df)} cores")
    
    return agg_df

def correlate_with_ihc(pred_df, ihc_df, gene_name, ihc_column, output_dir):
    """
    Correlate predicted gene expression with IHC scores.
    
    Args:
        pred_df: DataFrame with predicted expression (cores x genes)
        ihc_df: DataFrame with IHC scores
        gene_name: Gene to correlate (e.g., 'PAX5', 'CD19')
        ihc_column: Column in ihc_df with ground truth scores
        output_dir: Directory to save plots
    """
    # Check if gene exists in predictions
    if gene_name not in pred_df.columns:
        print(f"WARNING: Gene '{gene_name}' not found in predictions. Available genes: {pred_df.columns[:10].tolist()}...")
        return None
    
    # Merge predictions with IHC scores
    # Reset index to get core as column
    pred_merge = pred_df.reset_index()
    pred_merge = pred_merge.rename(columns={pred_merge.columns[0]: 'pred_core_id'})
    
    # Clean up core IDs for matching
    # Prediction core_ids might have format like "TMA3_..." or just the core number
    # IHC core_ids have format "1-XXX"
    pred_merge['pred_core_id_str'] = pred_merge['pred_core_id'].astype(str)
    ihc_df_copy = ihc_df.copy()
    ihc_df_copy['ihc_core_id_str'] = ihc_df_copy['core_id'].astype(str)
    
    print(f"  Prediction core_ids (first 10): {sorted(pred_merge['pred_core_id_str'].unique())[:10]}")
    print(f"  IHC core_ids (first 10): {sorted(ihc_df_copy['ihc_core_id_str'].unique())[:10]}")
    
    # Try direct match first
    merged = pd.merge(
        pred_merge[['pred_core_id_str', gene_name]],
        ihc_df_copy[['ihc_core_id_str', ihc_column]],
        left_on='pred_core_id_str',
        right_on='ihc_core_id_str',
        how='inner'
    )
    
    if len(merged) == 0:
        print(f"  No direct matches, trying to extract core numbers...")
        # Try extracting just the numeric part from both
        # For pred: might be "1", "2", "001", "002" etc
        # For IHC: "1-621" -> "621"
        ihc_df_copy['core_num'] = ihc_df_copy['ihc_core_id_str'].str.extract(r'-0*(\d+)$')[0]
        pred_merge['core_num'] = pred_merge['pred_core_id_str'].str.extract(r'0*(\d+)$')[0]
        
        print(f"    Extracted pred numbers: {sorted(pred_merge['core_num'].dropna().unique())[:10]}")
        print(f"    Extracted IHC numbers: {sorted(ihc_df_copy['core_num'].dropna().unique())[:10]}")
        
        merged = pd.merge(
            pred_merge[['core_num', gene_name]],
            ihc_df_copy[['core_num', ihc_column]],
            on='core_num',
            how='inner'
        )
    
    if len(merged) == 0:
        print(f"  No matches found with any strategy")
        return None
    
    # Drop NaN values
    merged = merged.dropna(subset=[gene_name, ihc_column])
    
    if len(merged) == 0:
        print(f"WARNING: No matching cores found for {gene_name} vs {ihc_column}")
        return None
    
    print(f"\nCorrelation: {gene_name} vs {ihc_column}")
    print(f"  N cores: {len(merged)}")
    
    # Calculate correlations
    pearson_r, pearson_p = pearsonr(merged[gene_name], merged[ihc_column])
    spearman_r, spearman_p = spearmanr(merged[gene_name], merged[ihc_column])
    
    print(f"  Pearson r: {pearson_r:.3f} (p={pearson_p:.3e})")
    print(f"  Spearman r: {spearman_r:.3f} (p={spearman_p:.3e})")
    
    return {
        'gene': gene_name,
        'ihc_column': ihc_column,
        'n_cores': len(merged),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'data': merged
    }

def plot_correlation(corr_result, aggregation, output_dir):
    """Create scatter plot for correlation."""
    if corr_result is None:
        return
    
    data = corr_result['data']
    gene = corr_result['gene']
    ihc_col = corr_result['ihc_column']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(data[gene], data[ihc_col], alpha=0.6, s=100)
    
    # Add regression line
    z = np.polyfit(data[gene], data[ihc_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(data[gene].min(), data[gene].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    # Labels and title
    ax.set_xlabel(f'Predicted {gene} Expression ({aggregation})', fontsize=12)
    ax.set_ylabel(f'IHC {ihc_col} H-score', fontsize=12)
    ax.set_title(
        f'{gene} Prediction vs IHC ({aggregation})\n'
        f'Pearson r={corr_result["pearson_r"]:.3f}, p={corr_result["pearson_p"]:.3e}\n'
        f'Spearman r={corr_result["spearman_r"]:.3f}, n={corr_result["n_cores"]}',
        fontsize=14
    )
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Save
    output_file = output_dir / f'{gene}_{ihc_col}_{aggregation}_correlation.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze TMA predictions vs IHC ground truth')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions h5ad file')
    parser.add_argument('--ihc', type=str, required=True,
                       help='Path to IHC ground truth Excel file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for plots and results')
    parser.add_argument('--std-threshold', type=float, default=0.1,
                       help='Standard deviation threshold for filtering genes (default: 0.1)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    adata = load_predictions(args.predictions)
    ihc_df = load_ihc_ground_truth(args.ihc)
    
    # Filter for variable genes
    adata_var = filter_variable_genes(adata, std_threshold=args.std_threshold)
    
    # Analyze both aggregation methods
    results = []
    
    for aggregation in ['mean', 'p90']:
        print(f"\n{'='*60}")
        print(f"Analyzing with {aggregation} aggregation")
        print(f"{'='*60}")
        
        # Aggregate by core
        agg_df = aggregate_by_core(adata, aggregation=aggregation)
        
        # Correlate PAX5
        if 'PAX5' in adata.var_names:
            pax5_result = correlate_with_ihc(agg_df, ihc_df, 'PAX5', 'PAX5_Hscore', output_dir)
            if pax5_result:
                results.append({**pax5_result, 'aggregation': aggregation})
                plot_correlation(pax5_result, aggregation, output_dir)
        
        # Correlate CD19
        if 'CD19' in adata.var_names:
            cd19_result = correlate_with_ihc(agg_df, ihc_df, 'CD19', 'CD19_Hscore', output_dir)
            if cd19_result:
                results.append({**cd19_result, 'aggregation': aggregation})
                plot_correlation(cd19_result, aggregation, output_dir)
    
    # Save results summary
    if results:
        results_df = pd.DataFrame([
            {
                'gene': r['gene'],
                'ihc_column': r['ihc_column'],
                'aggregation': r['aggregation'],
                'n_cores': r['n_cores'],
                'pearson_r': r['pearson_r'],
                'pearson_p': r['pearson_p'],
                'spearman_r': r['spearman_r'],
                'spearman_p': r['spearman_p']
            }
            for r in results
        ])
        
        results_file = output_dir / 'correlation_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\nSaved correlation results to {results_file}")
        print("\nSummary:")
        print(results_df)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")

if __name__ == '__main__':
    main()
