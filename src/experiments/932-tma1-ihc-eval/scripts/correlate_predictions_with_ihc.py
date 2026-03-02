"""
Correlate gene expression predictions with IHC ground truth.

This script:
1. Loads predicted gene expression from h5ad file
2. Loads TMA grid -> patient ID mapping
3. Aggregates predictions by grid position (mean and 90th percentile)
4. Correlates predicted PAX5 and CD19 expression with histopathologist IHC scores
5. Creates scatter plots showing correlations
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

def load_fov_to_patient_mapping(mapping_file):
    """Load TMA FOV to patient ID mapping."""
    print(f"Loading TMA FOV mapping from {mapping_file}...")
    df = pd.read_csv(mapping_file)
    # Ensure sample_id is string for merging
    df['sample_id'] = df['sample_id'].astype(str)
    print(f"Loaded mapping for {len(df)} FOVs ({df['grid_position'].nunique()} unique grid positions)")
    return df

def load_ihc_ground_truth(ihc_file):
    """Load IHC ground truth scores from Excel file."""
    print(f"Loading IHC ground truth from {ihc_file}...")
    df = pd.read_excel(ihc_file, header=1)
    
    # Extract sample_id from core_id
    df['sample_id'] = df['core_id'].str.extract(r'-0*(\d+)$')[0]
    
    # Keep only relevant columns
    df = df[['Blocks', 'core_id', 'sample_id', 'CD19', 'CD20 H-score', 'PAX5 H-score']].copy()
    df = df.rename(columns={
        'CD20 H-score': 'CD20_Hscore',
        'PAX5 H-score': 'PAX5_Hscore',
        'CD19': 'CD19_Hscore'
    })
    
    print(f"Loaded {len(df)} patient records")
    print(f"  CD19 scores: {df['CD19_Hscore'].notna().sum()} non-null")
    print(f"  CD20 scores: {df['CD20_Hscore'].notna().sum()} non-null")
    print(f"  PAX5 scores: {df['PAX5_Hscore'].notna().sum()} non-null")
    
    return df

def filter_variable_genes(adata, std_threshold=0.1):
    """Filter for genes with sufficient variability (non-boring genes)."""
    gene_std = np.std(adata.X, axis=0)
    if hasattr(gene_std, 'A1'):  # If sparse matrix
        gene_std = gene_std.A1
    
    variable_mask = gene_std > std_threshold
    n_variable = variable_mask.sum()
    
    print(f"Filtering genes with std > {std_threshold}")
    print(f"  Kept {n_variable}/{len(variable_mask)} genes ({100*n_variable/len(variable_mask):.1f}%)")
    
    if n_variable == 0:
        print(f"  WARNING: No genes pass threshold, using all genes instead")
        return adata
    
    return adata[:, variable_mask].copy()

def aggregate_by_core(adata, aggregation='mean'):
    """
    Aggregate expression by core (FOV).
    
    Args:
        adata: AnnData object with predictions
        aggregation: 'mean' or 'p90' (90th percentile)
    
    Returns:
        DataFrame with aggregated expression per FOV
    """
    # Find core identifier column
    if 'fov' in adata.obs.columns:
        core_column = 'fov'
    elif 'core_id' in adata.obs.columns:
        core_column = 'core_id'
    elif 'core' in adata.obs.columns:
        core_column = 'core'
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
    
    # Reset index to have fov as column
    agg_df = agg_df.reset_index()
    agg_df = agg_df.rename(columns={core_column: 'fov'})
    
    print(f"  Aggregated to {len(agg_df)} FOVs")
    
    return agg_df

def correlate_with_ihc(pred_df, mapping_df, ihc_df, gene_name, ihc_column, output_dir, aggregation):
    """
    Correlate predicted gene expression with IHC scores.
    
    Args:
        pred_df: DataFrame with predicted expression (FOVs x genes)
        mapping_df: DataFrame mapping FOV to sample_id (and grid_position)
        ihc_df: DataFrame with IHC scores (by sample_id)
        gene_name: Gene to correlate (e.g., 'PAX5', 'CD19')
        ihc_column: Column in ihc_df with ground truth scores
        output_dir: Directory to save plots
        aggregation: Aggregation method used ('mean' or 'p90')
    
    Returns:
        Dictionary with correlation results
    """
    # Check if gene exists in predictions
    if gene_name not in pred_df.columns:
        print(f"WARNING: Gene '{gene_name}' not found in predictions.")
        return None
    
    # Merge predictions with FOV mapping
    merged = pd.merge(
        pred_df[['fov', gene_name]],
        mapping_df[['fov', 'grid_position', 'sample_id']],
        on='fov',
        how='inner'
    )
    
    # Merge with IHC scores
    merged = pd.merge(
        merged,
        ihc_df[['sample_id', ihc_column]],
        on='sample_id',
        how='inner'
    )
    
    # Drop NaN values
    merged = merged.dropna(subset=[gene_name, ihc_column])
    
    if len(merged) == 0:
        print(f"WARNING: No matching FOVs found for {gene_name} vs {ihc_column}")
        return None
    
    print(f"\nCorrelation: {gene_name} vs {ihc_column} ({aggregation})")
    print(f"  N FOVs: {len(merged)}")
    print(f"  N unique cores: {merged['grid_position'].nunique()}")
    print(f"  N unique patients: {merged['sample_id'].nunique()}")
    
    # Calculate correlations
    pearson_r, pearson_p = pearsonr(merged[gene_name], merged[ihc_column])
    spearman_r, spearman_p = spearmanr(merged[gene_name], merged[ihc_column])
    
    print(f"  Pearson r: {pearson_r:.3f} (p={pearson_p:.3e})")
    print(f"  Spearman r: {spearman_r:.3f} (p={spearman_p:.3e})")
    
    # Create plot
    plot_correlation(merged, gene_name, ihc_column, pearson_r, pearson_p, 
                    spearman_r, len(merged), aggregation, output_dir)
    
    return {
        'gene': gene_name,
        'ihc_column': ihc_column,
        'aggregation': aggregation,
        'n_fovs': len(merged),
        'n_cores': merged['grid_position'].nunique(),
        'n_patients': merged['sample_id'].nunique(),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p
    }

def plot_correlation(data, gene, ihc_col, pearson_r, pearson_p, spearman_r, n_fovs, aggregation, output_dir):
    """Create scatter plot for correlation."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(data[gene], data[ihc_col], alpha=0.6, s=100, edgecolors='k', linewidths=0.5)
    
    # Add regression line
    z = np.polyfit(data[gene], data[ihc_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(data[gene].min(), data[gene].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    
    # Labels and title
    ax.set_xlabel(f'Predicted {gene} Expression ({aggregation})', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'IHC {ihc_col.replace("_", " ")}', fontsize=14, fontweight='bold')
    ax.set_title(
        f'{gene} Prediction vs IHC ({aggregation} aggregation)\n'
        f'Pearson r={pearson_r:.3f}, p={pearson_p:.3e} | '
        f'Spearman r={spearman_r:.3f} | n={n_fovs} FOVs',
        fontsize=14, fontweight='bold'
    )
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12)
    
    # Save
    output_file = output_dir / f'{gene}_{ihc_col}_{aggregation}_correlation.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Correlate TMA predictions with IHC ground truth')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions h5ad file')
    parser.add_argument('--mapping', type=str, required=True,
                       help='Path to TMA FOV to patient mapping CSV file (with fov, grid_position, sample_id columns)')
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
    mapping_df = load_fov_to_patient_mapping(args.mapping)
    ihc_df = load_ihc_ground_truth(args.ihc)
    
    # Filter for variable genes (optional, for general correlation plots)
    # adata_var = filter_variable_genes(adata, std_threshold=args.std_threshold)
    
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
            pax5_result = correlate_with_ihc(agg_df, mapping_df, ihc_df, 
                                            'PAX5', 'PAX5_Hscore', output_dir, aggregation)
            if pax5_result:
                results.append(pax5_result)
        
        # Correlate CD19
        if 'CD19' in adata.var_names:
            cd19_result = correlate_with_ihc(agg_df, mapping_df, ihc_df, 
                                            'CD19', 'CD19_Hscore', output_dir, aggregation)
            if cd19_result:
                results.append(cd19_result)
    
    # Save results summary
    if results:
        results_df = pd.DataFrame(results)
        
        results_file = output_dir / 'correlation_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\nSaved correlation results to {results_file}")
        print("\nSummary:")
        print(results_df)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")

if __name__ == '__main__':
    main()
