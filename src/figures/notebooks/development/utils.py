# Essential imports for the functions used in development_analysis.ipynb
import pandas as pd
import numpy as np
import warnings
import time
from pathlib import Path
from collections import defaultdict
from typing import Optional, Union
from Bio import Entrez  # For PubMed search
import gseapy as gp  # For enrichment analysis
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import re

def plot_mean_score_heatmap(df, title, filename, outdir,
                           cmap="RdBu_r", vmin=None, vmax=None,
                           row_cluster=False, col_cluster=False,
                           xlabel="Time Point", ylabel="Term",
                           cbar_title="Mean Score", figsize=(8, 3)):
    """Plots a clustered heatmap of mean scores."""
    data_to_plot = df.copy()


    if vmin is None:
            vmin = data_to_plot.min().min()
    if vmax is None:
            vmax = data_to_plot.max().max()
    abs_max = max(abs(vmin), abs(vmax))
    vmin = -abs_max
    vmax = abs_max

    g = sns.clustermap(data_to_plot.T, cmap=cmap, vmin=vmin, vmax=vmax,
                       row_cluster=row_cluster, col_cluster=col_cluster,
                       figsize=figsize, cbar_pos=(0.02, .3, .03, .4), # Adjust cbar position
                       linewidths=0.5, linecolor='gray') # Add subtle lines

    g.ax_heatmap.set_xlabel(xlabel)
    g.ax_heatmap.set_ylabel(ylabel)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=0)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    g.cax.set_title(cbar_title, rotation=0, ha='left', va='center', fontsize=10) # Adjust cbar title
    g.ax_heatmap.grid(False) # Turn off grid lines on heatmap
    plt.suptitle(title, y=1.02) # Add main title above heatmap

    #rotate the x ticks
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    filepath = outdir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to: {filepath}")
    plt.show()
    plt.close() # Close figure to free memory

def run_gsea_enrichment(
    gene_list: list[str],
    gene_sets: list[str],
    organism: str = 'human',
    outdir: Optional[Union[str, Path]] = None,
    verbose: bool = False
) -> dict[str, pd.DataFrame]:
    """
    Perform gene set enrichment analysis using gseapy.enrichr.

    Args:
        gene_list (list of str): List of gene symbols.
        gene_sets (list of str): List of Enrichr library names.
        organism (str): Organism name.
        outdir (str or Path, optional): Directory to save results.
        verbose (bool): Whether to print verbose output.

    Returns:
        dict: {library_name: DataFrame of results (empty if failed or no results)}
    """
    if not gene_list:
        print("Warning: Empty gene list provided for enrichment analysis.")
        return {}

    enr_df_dict = {}
    for library in gene_sets:
        try:
            enr_result = gp.enrichr(
                gene_list=gene_list,
                gene_sets=library,
                organism=organism,
                outdir=outdir, # If None, results are not written to disk
                cutoff=0.05, # Default p-value cutoff
                verbose=verbose
            )
            if enr_result is not None and not enr_result.results.empty:
                enr_df_dict[library] = enr_result.results
            else:
                print(f"Warning: No significant enrichment found for library {library}.")
                # Store an empty DataFrame to indicate no results
                enr_df_dict[library] = pd.DataFrame()
        except Exception as e:
            print(f"Error running enrichment for library {library}: {e}")
            enr_df_dict[library] = pd.DataFrame() # Store empty DataFrame on error
    return enr_df_dict


def search_pubmed(term, retmax=10000,PUBMED_RATE_LIMIT_DELAY=0.5):
    """Searches PubMed and returns a set of PMIDs."""
    pmids = set()
    try:
        handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax, usehistory='y', retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        pmids = set(records["IdList"])
        # print(f"Found {len(pmids)} PMIDs for term: {term}") # Verbose logging
    except Exception as e:
        warnings.warn(f"PubMed search failed for term '{term}': {e}")
    # Rate limiting
    time.sleep(PUBMED_RATE_LIMIT_DELAY)
    return pmids


def perform_literature_search(genes, organs, csv_path_lens, PUBMED_RATE_LIMIT_DELAY=0.5):
    """Performs PubMed search for CW, Ref, and overlapping markers."""

    literature_df = pd.DataFrame(index=genes, columns=[f"{organ}: Number of PMIDs" for organ in organs])

    print(f"Starting PubMed search ...")
    for organ in organs:

        print(f"Searching for {organ}")

        for i, gene in enumerate(genes):

            if i % 1000 == 0:
                print(f"Processing gene number {i}")
            term = f'{organ} AND {gene}'

            pmids= search_pubmed(term,PUBMED_RATE_LIMIT_DELAY=PUBMED_RATE_LIMIT_DELAY)

            literature_df.loc[gene, f"{organ}: Number of PMIDs"] = len(pmids)


    literature_df.to_csv(csv_path_lens)

    return literature_df


def get_highest_expression_day(adata):
    """
    Calculate mean expression of each gene per day and identify the day 
    with highest expression for each gene.
    
    Parameters:
    -----------
    adata : AnnData
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with gene names and their highest expression day
    """
    # Get unique days
    days = adata.obs["Timepoint (day)"].unique()
    
    # Initialize dictionary to store mean expression per day
    mean_expr_by_day = {}
    q97_expr_by_day ={}
    
    # Calculate mean expression for each day
    for day in days:
        # Get cells for this day
        day_mask = adata.obs['Timepoint (day)'] == day
        # Calculate mean expression across cells for this day
        mean_expr_by_day[day] = adata.X[day_mask].mean(axis=0)
        q97_expr_by_day[day] = np.quantile(adata.X[day_mask], 0.97, axis=0)
    
    # Convert to DataFrame for easier manipulation
    # Assuming var_names contains gene names
    gene_names = adata.var_names
    
    # Create DataFrame with genes as rows and days as columns
    mean_expr_df = pd.DataFrame({day: mean_expr_by_day[day] for day in days}, 
                               index=gene_names)
    highest_value = mean_expr_df.max(axis=1)
    # z-normalize it:
    mean_expr_df = (mean_expr_df - mean_expr_df.mean(axis=1).values[:, np.newaxis]) / mean_expr_df.std(axis=1).values[:, np.newaxis]
    
    q97_expr_df = pd.DataFrame({day: q97_expr_by_day[day] for day in days}, 
                               index=gene_names)  
    highest_q97_value= q97_expr_df.max(axis=1)
    # z-normalize it:
    q97_expr_df = (q97_expr_df - q97_expr_df.mean(axis=1).values[:, np.newaxis]) / q97_expr_df.std(axis=1).values[:, np.newaxis] 


    # out of d17.d18,d19,d20 and d21, find the one with the highest expression
    # get the index of the highest value
    CS8_mean_expr_zscore = mean_expr_df[["d17","d18","d19","d20","d21"]].max(axis=1)
    CS8_Q97_expr_zscore = q97_expr_df[["d17","d18","d19","d20","d21"]].max(axis=1)

    
    # Combine into result DataFrame
    result = pd.DataFrame({
        'gene': gene_names,
        'highest_mean_expression_day': mean_expr_df.idxmax(axis=1),
        'mean_expression_at_highest_day': highest_value,
        'zscored_mean_expression_at_highest_day': mean_expr_df.max(axis=1),

        'highest_Q97_expression_day': q97_expr_df.idxmax(axis=1),
        'Q97_expression_at_highest_day': highest_q97_value,
        'zscored_Q97_expression_at_highest_day': q97_expr_df.max(axis=1),

        'CS8_mean_expression_zscore': CS8_mean_expr_zscore,
        'CS8_Q97_expression_zscore': CS8_Q97_expr_zscore

    })
   
    return result
