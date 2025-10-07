"""
Generate UMAP plot of aggregated transcriptomes to assess biological signal preservation.

This script creates a UMAP visualization of the patch-level aggregated transcriptomes
to answer the question: Is there still meaningful biological signal after merging 
cells at the patch level?
"""

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Configure scanpy
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

# Load the processed data
adata = sc.read_h5ad(snakemake.input.adata)

print(f"Loaded data with {adata.n_obs} patches and {adata.n_vars} genes")
print(f"Data shape: {adata.X.shape}")

# Basic preprocessing for UMAP
# Copy to avoid modifying original data
adata_umap = adata.copy()

# Normalize and log transform
sc.pp.normalize_total(adata_umap, target_sum=1e4)
sc.pp.log1p(adata_umap)

# Find highly variable genes
sc.pp.highly_variable_genes(adata_umap, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata_umap.raw = adata_umap
adata_umap = adata_umap[:, adata_umap.var.highly_variable]

print(f"Using {adata_umap.n_vars} highly variable genes for UMAP")

# Scale data
sc.pp.scale(adata_umap, max_value=10)

# Principal component analysis
sc.tl.pca(adata_umap, svd_solver='arpack')

# Compute neighborhood graph
sc.pp.neighbors(adata_umap, n_neighbors=10, n_pcs=40)

# Compute UMAP
sc.tl.umap(adata_umap)

# Leiden clustering for visualization
sc.tl.leiden(adata_umap, resolution=0.5)

# Create the plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Transcriptome UMAP Analysis - {snakemake.wildcards.dataset}', fontsize=16)

# Plot 1: UMAP colored by Leiden clusters
sc.pl.umap(adata_umap, color='leiden', ax=axes[0,0], show=False, frameon=False)
axes[0,0].set_title('Leiden Clusters')

# Plot 2: UMAP colored by total UMI count
adata_umap.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
sc.pl.umap(adata_umap, color='total_counts', ax=axes[0,1], show=False, frameon=False)
axes[0,1].set_title('Total UMI Counts')

# Plot 3: UMAP colored by number of detected genes
adata_umap.obs['n_genes'] = np.array((adata.X > 0).sum(axis=1)).flatten()
sc.pl.umap(adata_umap, color='n_genes', ax=axes[1,0], show=False, frameon=False)
axes[1,0].set_title('Number of Detected Genes')

# Plot 4: UMAP colored by cell count (if available)
if 'cell_count' in adata.obs.columns:
    adata_umap.obs['cell_count'] = adata.obs['cell_count']
    sc.pl.umap(adata_umap, color='cell_count', ax=axes[1,1], show=False, frameon=False)
    axes[1,1].set_title('Cells per Patch')
else:
    # Alternative: color by spatial coordinates if available
    if 'x_pixel' in adata.obs.columns:
        adata_umap.obs['x_pixel'] = adata.obs['x_pixel']
        sc.pl.umap(adata_umap, color='x_pixel', ax=axes[1,1], show=False, frameon=False)
        axes[1,1].set_title('X Coordinate')
    else:
        axes[1,1].text(0.5, 0.5, 'No additional\nmetadata available', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Additional Metadata')

plt.tight_layout()

# Save the plot
plt.savefig(snakemake.output.umap_plot, dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
print("\n=== TRANSCRIPTOME AGGREGATION QUALITY ASSESSMENT ===")
print(f"Number of patches: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")
print(f"Number of highly variable genes: {adata_umap.n_vars}")
print(f"Number of Leiden clusters: {len(adata_umap.obs['leiden'].unique())}")

if 'cell_count' in adata.obs.columns:
    print(f"Cells per patch - Mean: {adata.obs['cell_count'].mean():.1f}, "
          f"Median: {adata.obs['cell_count'].median():.1f}, "
          f"Range: {adata.obs['cell_count'].min()}-{adata.obs['cell_count'].max()}")

print(f"Total UMI counts - Mean: {adata_umap.obs['total_counts'].mean():.0f}, "
      f"Median: {adata_umap.obs['total_counts'].median():.0f}")
print(f"Detected genes per patch - Mean: {adata_umap.obs['n_genes'].mean():.0f}, "
      f"Median: {adata_umap.obs['n_genes'].median():.0f}")

print("\nBiological signal assessment:")
print(f"- Distinct clusters formed: {len(adata_umap.obs['leiden'].unique())} clusters suggest preserved biological heterogeneity")
print(f"- Transcriptional diversity: {adata_umap.n_vars} highly variable genes indicate maintained expression patterns")
print("- UMAP structure: Check plot for meaningful clustering patterns vs. technical artifacts")
