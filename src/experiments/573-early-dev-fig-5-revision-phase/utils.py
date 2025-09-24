# ### 1.1 Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
import plotly.graph_objects as go
from IPython.core.display import display, HTML
from scipy import stats
from scipy.cluster import hierarchy as sch
from scipy.sparse import issparse # For checking sparse matrix type
import re
import json
import subprocess # Retained for potential non-API uses, but API calls removed
import shlex
import statsmodels.api as sm # Use main API instead of sandbox if possible
from statsmodels.stats.multitest import multipletests # Corrected import path
import scanpy as sc
import anndata
import scipy
from collections import defaultdict, Counter
import os
import time
import datetime
import zipfile
import pickle as pk
from matplotlib_venn import venn2
import networkx as nx
import warnings
from pathlib import Path # For better path handling
from SPARQLWrapper import SPARQLWrapper, JSON # External Call
from Bio import Entrez # External Call
import gseapy as gp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Optional, Union
# if module is available, import it
try:
    from statannotations.Annotator import Annotator # For significance annotations
    statsannotations_available = True
except ImportError:
    print("statannotations module not found. Skipping import.")
    statsannotations_available = False



def fishers_exact_test(setA, setB, background_size):
    """
    Performs Fisher's exact test for enrichment using the hypergeometric distribution.

    Args:
        setA (set): First set of items (e.g., significant genes).
        setB (set): Second set of items (e.g., pathway genes).
        background_size (int): Total number of possible items in the background/universe.

    Returns:
        float: The p-value (right-tail probability) from the hypergeometric test.
               Lower p-value indicates significant overlap.
    """
    M = background_size  # Population size
    n = len(setA)        # Number of successes in population (size of set A)
    N = len(setB)        # Sample size (size of set B)
    x = len(setA & setB) # Number of drawn successes (overlap)

    # Survival function (1 - cdf) is often preferred for enrichment p-values
    # sf(k, M, n, N) = P(X > k)
    # We want P(X >= x), which is sf(x-1, M, n, N)
    if x == 0: # Handle edge case where overlap is zero
         return 1.0
    return stats.hypergeom.sf(x - 1, M, n, N)

# POTENTIAL ISSUE: SPARQLWrapper makes an external network call.
# Consider fetching this once and saving to a local file (e.g., JSON, CSV)
# for reproducible and offline analysis.
def fetch_hsapdv_ontology_sparql(endpoint_url="http://sparql.hegroup.org/sparql/"):
    """
    Fetches Human Skeletal Automated Phenotype Detection Vocabulary (HsapDv)
    ontology terms and definitions via SPARQL query.

    Args:
        endpoint_url (str): The SPARQL endpoint URL.

    Returns:
        dict: Dictionary mapping HsapDv term labels to their definitions.
              Returns empty dict on failure.
    """
    print("Attempting to fetch HsapDv ontology via SPARQL...")
    ontology_dict = {}
    try:
        sparql = SPARQLWrapper(endpoint_url)
        sparql.setQuery("""
        PREFIX obo: <http://purl.obolibrary.org/obo/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?term ?label ?definition
        WHERE {
            # Adjust the root term if necessary based on the ontology structure
            ?term rdfs:subClassOf* obo:HsapDv_0000000 . # Subclasses of the root
            ?term rdfs:label ?label .
            OPTIONAL { ?term obo:IAO_0000115 ?definition } # Definitions
        }
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        for result in results["results"]["bindings"]:
            term_label = result["label"]["value"]
            definition = result.get("definition", {}).get("value", "No description available.")
            ontology_dict[term_label] = definition
        print(f"Fetched {len(ontology_dict)} ontology terms.")
    except Exception as e:
        print(f"Error fetching ontology via SPARQL: {e}")
        print("Proceeding without ontology data.")
        # POTENTIAL ISSUE: Execution continues without ontology data. Consider halting or using cached data.
    return ontology_dict

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


def plot_mean_score_heatmap(df, title, filename, outdir,
                           cmap="RdBu_r", vmin=None, vmax=None,
                           row_cluster=False, col_cluster=False,
                           xlabel="Time Point", ylabel="Term",
                           cbar_title="Mean Score", figsize=(8, 3),
                           normalize=None):
    """Plots a clustered heatmap of mean scores."""
    data_to_plot = df.copy()

    if normalize == 'row':
        # Row-wise Z-score normalization (rows=terms)
        scaler = StandardScaler()
        data_to_plot = pd.DataFrame(scaler.fit_transform(data_to_plot),
                                    index=data_to_plot.index, columns=data_to_plot.columns)
        if vmin is None: vmin = -2
        if vmax is None: vmax = 2
        cbar_title = "Row Z-Score\n" + cbar_title
    if normalize == 'col':
        # Row-wise Z-score normalization (rows=terms)
        scaler = StandardScaler()
        data_to_plot = pd.DataFrame(scaler.fit_transform(data_to_plot.T).T,
                                    index=data_to_plot.index, columns=data_to_plot.columns)
        if vmin is None: vmin = -2
        if vmax is None: vmax = 2
        cbar_title = "Col Z-Score\n" + cbar_title
    elif normalize == 'zscore':
        # Z-score based on global mean/std per term (calculated elsewhere)
        # Assumes input df already contains z-scores
        if vmin is None: vmin = -4
        if vmax is None: vmax = 4
        cbar_title = "Z-Score\n" + cbar_title
    else: # Absolute scores
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
                       linewidths=0, linecolor='gray') # Add subtle lines

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

def plot_umap(adata, color_key, outdir, filename_prefix, basis="X_cellwhisperer_umap", **kwargs):
    """Plots a UMAP embedding."""
    fig, ax = plt.subplots() # Create figure and axes explicitly
    sc.pl.embedding(adata, basis=basis, color=color_key, show=False, ax=ax, **kwargs)
    ax.set_title(f"UMAP colored by {color_key}") # Set title on the axes
    ax.patch.set_facecolor('white') # Ensure white background
    plt.grid(False) # Ensure no grid
    filepath = outdir / f"{filename_prefix}_{color_key}_umap.pdf"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved UMAP to: {filepath}")
    # plt.show() # Avoid showing inline for many plots
    plt.close(fig) # Close figure

def calculate_overlap_stats(setA, setB, background_size):
    """Calculates overlap statistics (pval, Jaccard, LogOddsRatio)."""
    stats_dict = {}
    overlap = len(setA & setB)
    union = len(setA | setB)
    n_A = len(setA)
    n_B = len(setB)

    stats_dict["pval"] = fishers_exact_test(setA, setB, background_size)
    stats_dict["N_A"] = n_A
    stats_dict["N_B"] = n_B
    stats_dict["N_overlap"] = overlap
    stats_dict["Jaccard"] = overlap / union if union > 0 else 0

    # Calculate Log Odds Ratio
    # Expected overlap = (n_A * n_B) / background_size
    expected_overlap = (n_A * n_B) / background_size
    if overlap > 0 and expected_overlap > 0:
        # Add pseudocount for stability if necessary, though usually handled by Fisher's
        odds_ratio = (overlap / (n_B - overlap + 1e-9)) / \
                     ((n_A - overlap + 1e-9) / (background_size - n_A - n_B + overlap + 1e-9))
        # Simplified approximation (often used): log2(observed / expected)
        # log_odds_ratio = np.log2(overlap / expected_overlap)
        log_odds_ratio = np.log2(odds_ratio)

    elif overlap > 0 and expected_overlap == 0:
         log_odds_ratio = np.inf # Infinite enrichment
    else: # overlap == 0
        log_odds_ratio = -np.inf if expected_overlap > 0 else 0 # Infinite depletion or zero

    stats_dict["LogOddsRatio"] = log_odds_ratio

    return stats_dict

# --- Validation: Overlap with BBI Reference Markers ---

def validate_and_plot_markers(cw_marker_dict, ref_marker_dict, background_size, outdir, method_suffix,ENRICHMENT_LIBRARIES):
    """Calculates overlap, plots stats, Venn diagrams, and runs enrichment."""
    if not cw_marker_dict:
         print(f"Skipping validation for Method {method_suffix}: No CW markers found.")
         return None, None # Return None for overlaps and diffs

    all_overlap_stats = defaultdict(dict)
    all_venn_data = {}
    cw_only_markers = {}

    organs_to_validate = sorted(list(cw_marker_dict.keys())) # Use organs found by CW method

    pvals = []
    keys_for_fdr = []

    for organ in organs_to_validate:
        set_cw = set(cw_marker_dict[organ])
        set_ref = ref_marker_dict.get(organ, set()) # Get reference set, default to empty if missing

        if not set_cw: # Skip if CW set is empty for this organ
            continue

        overlap_stats = calculate_overlap_stats(set_cw, set_ref, background_size)
        all_overlap_stats[organ] = overlap_stats
        pvals.append(overlap_stats["pval"])
        keys_for_fdr.append(organ)

        all_venn_data[organ] = (set_cw, set_ref)
        cw_only_markers[organ] = set_cw - set_ref

    # Apply FDR correction
    if pvals:
        fdrs = adjust_fdr_bh(pvals)
        for i, organ in enumerate(keys_for_fdr):
            all_overlap_stats[organ]["fdr"] = fdrs[i]
    else:
        print("No p-values to adjust for FDR.")


    # --- Plot Overlap Statistics ---
    overlap_df = pd.DataFrame(all_overlap_stats).T # Transpose for organs as rows

    if not overlap_df.empty:
        overlap_df.to_csv(outdir / f"gene_overlap_stats_{method_suffix}.csv")
        print(f"Saved overlap statistics to: {outdir / f'gene_overlap_stats_{method_suffix}.csv'}")

        for y_col in ["Jaccard", "LogOddsRatio"]:
            if y_col in overlap_df.columns:
                plt.figure(figsize=(max(6, 0.5 * len(overlap_df)), 5)) # Adjust width based on number of organs
                plt.bar(overlap_df.index, overlap_df[y_col], color="grey")

                # Add significance stars based on FDR
                for i, organ in enumerate(overlap_df.index):
                    fdr = overlap_df.loc[organ, "fdr"]
                    y_pos = overlap_df.loc[organ, y_col]
                    stars = ""
                    if pd.notna(fdr):
                        if fdr < 0.001: stars = "***"
                        elif fdr < 0.01: stars = "**"
                        elif fdr < 0.05: stars = "*"
                    # Adjust text position slightly above the bar
                    plt.text(i, y_pos + 0.02 * abs(y_pos), stars, fontsize=12, ha="center")

                plt.xticks(rotation=90)
                plt.ylabel(y_col)
                plt.title(f"Overlap between CW Method {method_suffix} and Reference Markers")
                plot_path = outdir / f"gene_overlap_{y_col}_{method_suffix}.pdf"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Saved overlap plot to: {plot_path}")
                plt.show()
                plt.close()

    # --- Plot Venn Diagrams ---
    venn_dir = outdir / f"venn_diagrams_{method_suffix}"
    venn_dir.mkdir(exist_ok=True)
    for organ, (set_cw, set_ref) in all_venn_data.items():
        plt.figure(figsize=(4, 4))
        venn2([set_cw, set_ref], set_labels=(f'CW M{method_suffix}', 'Reference'))
        plt.title(organ.capitalize())
        plot_path = venn_dir / f"venn_{organ}_{method_suffix}.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
    print(f"Saved Venn diagrams to: {venn_dir}")


    # --- Enrichment Analysis of CW-Only Genes ---
    print(f"\n--- Enrichment Analysis for CW-Only Genes (Method {method_suffix}) ---")
    enrichment_results = {}
    enrichment_dir = outdir / f"enrichment_plots_{method_suffix}"
    enrichment_dir.mkdir(exist_ok=True)

    for organ, geneset in cw_only_markers.items():
         if len(geneset) > 5: # Only run enrichment if there are enough genes
             print(f"Running enrichment for {len(geneset)} CW-only genes in {organ}...")
             enr_dict = run_gsea_enrichment(list(geneset), ENRICHMENT_LIBRARIES, organism='human')

             # Combine results from different libraries if needed
             valid_dfs = [df for df in enr_dict.values() if not df.empty]
             if valid_dfs:
                 combined_enr_df = pd.concat(valid_dfs, ignore_index=True)
                 combined_enr_df = combined_enr_df.sort_values("Adjusted P-value")
                 enrichment_results[organ] = combined_enr_df

                 # Plotting (only top terms)
                 try:
                    gp.dotplot(combined_enr_df,
                            column="Adjusted P-value",
                            title=f"{organ.capitalize()} - Top Enriched Terms (CW M{method_suffix} Only)",
                            ofname=str(enrichment_dir / f"{organ}_enrichment_dotplot_{method_suffix}.pdf"), # Ensure path is string
                            top_term=10, # Show top 10 terms
                            figsize=(6, 5),
                            marker='o', show_ring=False)
                 except Exception as plot_err: # Catch potential plotting errors from gseapy
                     warnings.warn(f"Could not generate dotplot for {organ} (Method {method_suffix}): {plot_err}")

                # plot again, but as barplot

                
             else:
                 print(f"No significant enrichment found for CW-only genes in {organ}.")
         else:
             print(f"Skipping enrichment for {organ}: Too few CW-only genes ({len(geneset)}).")

    print(f"Saved enrichment plots to: {enrichment_dir}")
    return overlap_df, cw_only_markers


def get_developmental_diseases_dict(DISEASE_DATA_DIR,DISEASE_ZIP,PEDIATRIC_CANCER_GENES_PATH,adata):
    # Unzip the file first if neccesary
    if not DISEASE_DATA_DIR.exists():
        with zipfile.ZipFile(DISEASE_ZIP, 'r') as zip_ref:
            zip_ref.extractall(DISEASE_ZIP.parent)

    # --- Load Disease Gene Sets ---
    developmental_diseases_dict = {}
    disease_files = {
        "bone_development": "bone_development_disease.tab",
        "congenital_heart": "congenital_heart_disease.tab",
        "congenital_myopathy": "congenital_myopathy.tab",
        "fetal_akinesia": "fetal_akinesia.tab",
        "fetal_diseases": "fetal_diseases.tab",
        "newborn_diseases": "infant,newborn_diseases.tab",
        "language_development": "language_development.tab",
        "neurodevelopmental": "neurodevelopmental_disorders.tab",
    }
    available_genes = set(adata.var_names)
    for disease_key, filename in disease_files.items():
        filepath = DISEASE_DATA_DIR / filename
        try:
            # POTENTIAL ISSUE: Encoding might vary. Added error handling.
            df = pd.read_csv(filepath, sep="\t", encoding='utf-8', on_bad_lines='skip') # Try utf-8 first
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(filepath, sep="\t", encoding='latin-1', on_bad_lines='skip') # Fallback encoding
            except Exception as e:
                warnings.warn(f"Could not read disease file {filepath}: {e}")
                continue

        if 'Symbol' in df.columns:
            # Convert symbols to string, handle potential NaN, filter by available genes
            disease_genes = set(df['Symbol'].astype(str).dropna()) & available_genes
            developmental_diseases_dict[disease_key] = disease_genes
            print(f"Loaded {len(disease_genes)} genes for {disease_key} (available in data).")
        else:
            warnings.warn(f"'Symbol' column not found in {filepath}. Cannot load genes for {disease_key}.")
    # Load pediatric cancer genes
    try:
        ped_cancer_df = pd.read_csv(PEDIATRIC_CANCER_GENES_PATH, sep="\t")
        if 'Name' in ped_cancer_df.columns:
            ped_cancer_genes = set(ped_cancer_df['Name'].astype(str).dropna()) & available_genes
            developmental_diseases_dict['pediatric_cancer'] = ped_cancer_genes
            print(f"Loaded {len(ped_cancer_genes)} pediatric cancer genes (available in data).")
        else:
            warnings.warn(f"'Name' column not found in {PEDIATRIC_CANCER_GENES_PATH}.")
    except FileNotFoundError:
        warnings.warn(f"Pediatric cancer gene file not found at {PEDIATRIC_CANCER_GENES_PATH}.")
    except Exception as e:
        warnings.warn(f"Could not read pediatric cancer gene file: {e}")
        
    return developmental_diseases_dict

def perform_literature_search(genes, organs, csv_path_lens, csv_paths_pmids,PUBMED_RATE_LIMIT_DELAY=0.5):
    """Performs PubMed search for CW, Ref, and overlapping markers."""

    literature_df = pd.DataFrame(index=genes, columns=[f"{organ}: Number of PMIDs" for organ in organs])
    literature_df_w_pmids = pd.DataFrame(index=genes, columns=[f"{organ}: PMIDs" for organ in organs])

    print(f"Starting PubMed search ...")
    for organ in organs:

        print(f"Searching for {organ}")

        for i, gene in enumerate(genes):

            if i % 1000 == 0:
                print(f"Processing gene number {i}")
            #  term = f'"{organ}"[Title/Abstract] AND "{gene}"[Title/Abstract] AND "human"[MeSH Terms]' # More specific query
            term = f'{organ} AND {gene}'

            pmids= search_pubmed(term,PUBMED_RATE_LIMIT_DELAY=PUBMED_RATE_LIMIT_DELAY)

            literature_df.loc[gene, f"{organ}: Number of PMIDs"] = len(pmids)
            literature_df_w_pmids.loc[gene, f"{organ}: PMIDs"] = pmids


    literature_df.to_csv(csv_path_lens)
    literature_df_w_pmids.to_csv(csv_paths_pmids)

    return literature_df, literature_df_w_pmids


    # # Save results
    # with open(pk_path, 'wb') as handle:
    #     pk.dump(literature_results, handle, protocol=pk.HIGHEST_PROTOCOL)
    # print(f"Saved literature search results to: {pk_path}")
    # return literature_results

# def perform_literature_search(cw_marker_dict, ref_marker_dict, organs, outdir, method_suffix):
#     """Performs PubMed search for CW, Ref, and overlapping markers."""
#     if not cw_marker_dict:
#         print(f"Skipping literature search for Method {method_suffix}: No CW markers.")
#         return None

#     literature_results = {
#         "CW_Only": defaultdict(set),
#         "Reference_Only": defaultdict(set),
#         "Intersection": defaultdict(set),
#         "CW_All": defaultdict(set), # All genes identified by CW method
#         "Reference_All": defaultdict(set) # All genes from reference list
#     }

#     print(f"Starting PubMed search for Method {method_suffix} markers...")
#     for organ in organs:
#         if organ not in cw_marker_dict:
#              print(f"Skipping PubMed for {organ} (Method {method_suffix}): No CW markers found.")
#              continue

#         set_cw = set(cw_marker_dict[organ])
#         set_ref = ref_marker_dict.get(organ, set())
#         set_intersect = set_cw & set_ref
#         set_cw_only = set_cw - set_ref
#         set_ref_only = set_ref - set_cw

#         print(f"Searching for {organ}: CW_Only={len(set_cw_only)}, Ref_Only={len(set_ref_only)}, Intersect={len(set_intersect)}")

#         for gene in set_cw_only:
#             #  term = f'"{organ}"[Title/Abstract] AND "{gene}"[Title/Abstract] AND "human"[MeSH Terms]' # More specific query
#              term = f'{organ} AND {gene}'
#              literature_results["CW_Only"][(organ, gene)] = search_pubmed(term)
#              literature_results["CW_All"][(organ, gene)] = literature_results["CW_Only"][(organ, gene)] # Add to CW_All as well

#         for gene in set_ref_only:
#             #  term = f'"{organ}"[Title/Abstract] AND "{gene}"[Title/Abstract] AND "human"[MeSH Terms]'
#              term = f'{organ} AND {gene}'
#              literature_results["Reference_Only"][(organ, gene)] = search_pubmed(term)
#              literature_results["Reference_All"][(organ, gene)] = literature_results["Reference_Only"][(organ, gene)]

#         for gene in set_intersect:
#             #  term = f'"{organ}"[Title/Abstract] AND "{gene}"[Title/Abstract] AND "human"[MeSH Terms]'
#              term = f'{organ} AND {gene}'
#              pmids = search_pubmed(term)
#              literature_results["Intersection"][(organ, gene)] = pmids
#              # Add intersection results to both CW_All and Reference_All
#              literature_results["CW_All"][(organ, gene)] = pmids
#              literature_results["Reference_All"][(organ, gene)] = pmids


#     # Save results
#     pk_path = outdir / f'literature_search_results_{method_suffix}.pickle'
#     with open(pk_path, 'wb') as handle:
#         pk.dump(literature_results, handle, protocol=pk.HIGHEST_PROTOCOL)
#     print(f"Saved literature search results to: {pk_path}")
#     return literature_results

# --- Plot Literature Results ---
def plot_literature_comparison(lit_results,  outdir):
    """Compares literature counts between CW and Reference."""

    # Combine results into a plottable DataFrame
    plot_data = []
    sources = lit_results.copy() # Copy to avoid modifying original

    for method, data_dict in sources.items():
        for (organ, gene), pmids in data_dict.items():
             plot_data.append({
                 "Method": method,
                 "Organ": organ,
                 "Gene": gene,
                 "Num_Papers": len(pmids)
             })

    lit_df = pd.DataFrame(plot_data)

    # Handle log scale: add small value before log, or filter zeros
    lit_df["Num_Papers_Log"] = np.log10(lit_df["Num_Papers"] + 1) # Log(N+1) to handle zeros

    # --- Boxplot per Organ ---
    plt.figure(figsize=(14, 7))
    ax = sns.boxplot(data=lit_df, x='Organ', y='Num_Papers_Log', hue='Method', whis=[5, 95])
    plt.title('Literature Support for Identified Marker Genes (Log10(PMIDs + 1))')
    plt.xlabel('Organ')
    plt.ylabel('Log10(Number of PubMed IDs + 1)')
    plt.xticks(rotation=45, ha='right')

    # Add significance annotations
    pairs = []
    for organ in lit_df['Organ'].unique():
        pairs.append(((organ, "CW_All"), (organ, "Reference")))
        pairs.append(((organ, "CW_All"), (organ, "Intersection")))
        pairs.append(((organ, "Reference"), (organ, "Intersection")))


    if pairs and statsannotations_available:
         # POTENTIAL ISSUE: Using Mann-Whitney U test is generally safer for non-normal count data.
         annotator = Annotator(ax, pairs, data=lit_df, x='Organ', y='Num_Papers_Log', hue='Method')
         annotator.configure(test='Mann-Whitney', text_format='star', loc='outside', verbose=0)
         try:
              annotator.apply_and_annotate()
         except Exception as anno_err: # Catch potential errors in annotation library
              warnings.warn(f"Could not add significance annotations to literature plot: {anno_err}")


    plt.legend(title='Gene Set Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    filepath = outdir / "literature_comparison_per_organ_boxplot.pdf"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved literature comparison boxplot to: {filepath}")
    plt.show()
    plt.close()

    # --- Barplot Aggregated (showing means per organ) ---

    per_organ_oper_method_mean_citation_numbers=lit_df.groupby(["Method", "Organ"]).agg({"Num_Papers": "mean"})
    # plot as barplot with error bars
    means_per_method = per_organ_oper_method_mean_citation_numbers.groupby("Method").mean().Num_Papers.values
    sems_per_method = per_organ_oper_method_mean_citation_numbers.groupby("Method").sem().Num_Papers.values
    methods= per_organ_oper_method_mean_citation_numbers.groupby("Method").mean().index


    print(f"Means per method: {means_per_method}")
    print(f"SEMs per method: {sems_per_method}")
    print(f"Methods: {methods}")
    # plot:
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, means_per_method, yerr=sems_per_method, capsize=5, color="royalblue", alpha=0.7, edgecolor="black")
    plt.xlabel("Methods")
    plt.ylabel("Mean Number of Papers per Organ")
    plt.title("Bar Plot of Dictionary Values")
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    plt.show()

    # now boxplot
    sns.boxplot(data=per_organ_oper_method_mean_citation_numbers.reset_index(), x="Method", y="Num_Papers")
    plt.ylabel("Mean Number of Papers per Organ")
    plt.show()

    sns.boxplot(data=lit_df, x="Method", y="Num_Papers_Log")
    plt.ylabel("Log10(Number of papers + 1)")
    plt.show()





    # --- Barplot Aggregated ---
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=lit_df, x='Method', y='Num_Papers_Log', estimator=np.mean, errorbar='se', capsize=0.1)
    plt.title('Average Literature Support per Method (Log10(PMIDs + 1))')
    plt.xlabel('Gene Set Source')
    plt.ylabel('Mean Log10(Number of PubMed IDs + 1) +/- SEM')

    # Add significance annotations
    agg_pairs = []
    methods = lit_df['Method'].unique()
    agg_pairs.append(("CW_All", "Reference"))
    agg_pairs.append(("Intersection", "Reference"))
    agg_pairs.append(("CW_All", "Intersection"))

    if agg_pairs and statsannotations_available:
        annotator_agg = Annotator(ax, agg_pairs, data=lit_df, x='Method', y='Num_Papers_Log')
        annotator_agg.configure(test='Mann-Whitney', text_format='star', loc='outside', verbose=0)
        try:
            annotator_agg.apply_and_annotate()
        except Exception as anno_err:
             warnings.warn(f"Could not add significance annotations to aggregated literature plot: {anno_err}")

    filepath = outdir / "literature_comparison_aggregated_barplot.pdf"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved aggregated literature barplot to: {filepath}")
    plt.show()
    plt.close()

    return lit_df

    

def adjust_fdr_bh(pvals):
    """
    Adjusts p-values using the Benjamini/Hochberg FDR correction.

    Args:
        pvals (list or np.array): List of p-values.

    Returns:
        np.array: Array of FDR-adjusted p-values.
    """
    if not pvals:
        return np.array([])
    return multipletests(pvals, method='fdr_bh')[1]


# --- Perform Enrichment Tests ---
def run_disease_enrichment(marker_gene_dict, disease_gene_dict, background_size):
    """Runs Fisher's exact test for disease enrichment in marker sets."""
    enrichment_results = {}
    pvals = []
    keys = []
    overlaps = defaultdict(list)

    for disease_name, disease_genes in disease_gene_dict.items():
        if not disease_genes: continue # Skip if no genes for this disease
        for organ, marker_genes in marker_gene_dict.items():
            if not marker_genes: continue # Skip if no marker genes for this organ

            # Ensure sets contain strings
            set_disease = set(map(str, disease_genes))
            set_marker = set(map(str, marker_genes))

            pval = fishers_exact_test(set_disease, set_marker, background_size)
            enrichment_results[(disease_name, organ)] = pval
            pvals.append(pval)
            keys.append((disease_name, organ))

            this_overlap= set_disease & set_marker
            for overlap_gene in this_overlap:
                overlaps[organ].append((overlap_gene, disease_name))
    # FDR correction
    if pvals:
         fdrs = adjust_fdr_bh(pvals)
         for i, key in enumerate(keys):
             enrichment_results[key] = {"pval": pvals[i], "fdr": fdrs[i]}
    else: # Add FDR column even if empty
        for key in enrichment_results:
             enrichment_results[key] = {"pval": enrichment_results[key], "fdr": np.nan}

    # turn overlaps into a dataframe
    for organ in overlaps.keys():
        overlaps[organ] = pd.DataFrame(overlaps[organ], columns=["Gene", "Disease"])
        # set Gene as index
        overlaps[organ].set_index("Gene", inplace=True)


    return pd.DataFrame.from_dict(enrichment_results, orient='index'), overlaps

# --- Plot Significant Results ---
def plot_significant_enrichment(enrichment_df, title, filename, outdir, fdr_thresh=0.05):
    """Plots significant disease enrichments as a heatmap."""
    if enrichment_df.empty:
        print(f"Skipping enrichment plot for {title}: No results.")
        return

    significant_df = enrichment_df[enrichment_df['fdr'] < fdr_thresh].copy()

    if significant_df.empty:
        print(f"No significant disease enrichments found for {title} at FDR < {fdr_thresh}.")
        return

    # Convert p-values to -log10 for visualization
    significant_df['-log10(FDR)'] = -np.log10(significant_df['fdr'] + 1e-10) # Add small epsilon

    # Pivot for heatmap (Disease x Organ)
    heatmap_df = significant_df.reset_index().pivot(index='level_0', columns='level_1', values='-log10(FDR)')
    heatmap_df = heatmap_df.fillna(0) # Fill non-significant pairs with 0

    plt.figure(figsize=(max(8, 0.6 * heatmap_df.shape[1]), max(6, 0.4 * heatmap_df.shape[0]))) # Adjust size
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5, cbar_kws={'label': '-log10(FDR)'})
    plt.title(title + f" (FDR < {fdr_thresh})")
    plt.xlabel("Organ")
    plt.ylabel("Disease Category")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    filepath = outdir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved significant disease enrichment heatmap to: {filepath}")
    plt.show()
    plt.close()

# --- Sankey Plot Comparison (Example: CW M1 vs Reference) ---
def plot_sankey_comparison(adata, col1, col2, title, filename, outdir):
    """Generates a Sankey plot comparing two dominant organ classifications."""
    if col1 not in adata.obs.columns or col2 not in adata.obs.columns:
         warnings.warn(f"Skipping Sankey plot: Columns {col1} or {col2} missing.")
         return

    # Filter out "None" assignments for comparison
    valid_mask = (adata.obs[col1] != "None") & (adata.obs[col2] != "None")
    if not valid_mask.any():
         print("Skipping Sankey plot: No cells with valid assignments in both columns.")
         return

    filtered_obs = adata.obs.loc[valid_mask, [col1, col2]]

    # Get unique organs/labels from both columns
    labels1 = sorted(filtered_obs[col1].unique())
    labels2 = sorted(filtered_obs[col2].unique())
    all_labels = labels1 + labels2
    label_indices = {label: i for i, label in enumerate(all_labels)}
    num_labels1 = len(labels1)

    # Map labels to indices
    source_indices = filtered_obs[col1].map(label_indices).tolist()
    target_indices = filtered_obs[col2].map(lambda x: label_indices[x]).tolist() # Map labels from col2

    # Count transitions
    link_counts = Counter(zip(source_indices, target_indices))
    sankey_source = [s for s, t in link_counts.keys()]
    sankey_target = [t for s, t in link_counts.keys()]
    sankey_value = list(link_counts.values())

    # Get colors for nodes
    node_colors = [ORGAN_COLORS.get(label, 'lightgrey') for label in all_labels]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=all_labels,
            color=node_colors
        ),
        link=dict(
            source=sankey_source,
            target=sankey_target,
            value=sankey_value,
            color="rgba(150, 150, 150, 0.4)" # Link color
        )
    ))
    fig.update_layout(title_text=title, font_size=10)
    filepath = outdir / filename
    try:
        fig.write_image(filepath, scale=2) # Increase scale for better resolution
        print(f"Saved Sankey plot to: {filepath}")
    except Exception as e:
         warnings.warn(f"Failed to save Sankey plot {filepath}. Ensure plotly-orca or kaleido is installed. Error: {e}")
    fig.show()

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

# Also plot the correlation between effect size of marker enrichment and literature support
def plot_and_get_enrichment_vs_literature(enrichment_results, literature_results, outdir, metric="scores", color_cutoff=2,
                                           color_cutoff_col=None, lowess=False, bins_edges = None, start_box=(4.5, -0.5), width_box=3,
):

    if bins_edges is None:
        bins_edges = [merged_df_this_organ[metric].min(),0,0.5, 1,1.5,2,2.5,3,merged_df_this_organ[metric].max()]

    merged_dfs_per_organ = {}
    if color_cutoff_col is None:
        color_cutoff_col = metric

    cw_lit_hit_numbers_df = literature_results[literature_results["Method"]=="CW_All"]

    fig, axes = plt.subplots(2,cw_lit_hit_numbers_df.Organ.nunique(), figsize=(cw_lit_hit_numbers_df.Organ.nunique()*4, 9), sharey=False)
    if type(axes) != np.ndarray:
        axes = [axes]
    # flatten axes if needed
    if len(axes.shape) == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i,organ in enumerate(cw_lit_hit_numbers_df.Organ.unique()):

        plt.sca(axes[i])

        # intersect with enrichment results

        merged_df_this_organ = enrichment_results[organ].merge(cw_lit_hit_numbers_df[cw_lit_hit_numbers_df["Organ"]==organ], left_on="names", right_on="Gene")
        merged_df_this_organ["Num_Papers_Log1p"] = np.log1p(merged_df_this_organ["Num_Papers"])

        sns.regplot(x=merged_df_this_organ[metric], y=merged_df_this_organ["Num_Papers_Log1p"], scatter=False,ax=plt.gca(),line_kws={"color":"black"}, lowess=lowess)
        sns.scatterplot(data=merged_df_this_organ[merged_df_this_organ[color_cutoff_col] >= color_cutoff], x=metric,
                        y="Num_Papers_Log1p", alpha=0.3,ax=plt.gca(),size=0.2)
        sns.scatterplot(data=merged_df_this_organ[merged_df_this_organ[color_cutoff_col] < color_cutoff], x=metric,
                        y="Num_Papers_Log1p", color="grey", alpha=0.3,ax=plt.gca(),size=0.2)
        sns.kdeplot(data=merged_df_this_organ, x=metric,
                    y="Num_Papers_Log1p", color="black", alpha=0.3,ax=plt.gca(), cut=True)
        
        # Add quadrant lines
        plt.axvline(x=color_cutoff, color='coral', linestyle='--')
        plt.axhline(y=np.log1p(0.5), color='coral', linestyle='--')

        plt.xlabel(metric)
        plt.ylabel("Log10(Number of Papers + 1)")
        plt.tight_layout()

        # Annotate the top 5 genes with highest enrichment strength but with low literature support
        top_genes = merged_df_this_organ[merged_df_this_organ["Num_Papers"]==0].nlargest(5, metric)
        for _, row in top_genes.iterrows():
            plt.annotate(row["names"], (row[metric], np.log1p(row["Num_Papers"])), fontsize=2, color="red", ha='center', va='bottom')
            


        # print correlation coefficient and pvalue
        r, pval = stats.pearsonr(merged_df_this_organ[metric], merged_df_this_organ["Num_Papers_Log1p"])
        plt.title(f"Enrichment Strength vs Literature Support\nfor {organ}\nr={r:.2f}, p={pval:.2e}")

        plt.legend([],[], frameon=False)



        #### Second plot: Violin plots over bins of enrichment strength

        plt.sca(axes[cw_lit_hit_numbers_df.Organ.nunique()+i])

        merged_df_this_organ["bin"] = pd.cut(merged_df_this_organ[metric], bins=bins_edges)
        merged_df_this_organ["above_color_cutoff"] = merged_df_this_organ[color_cutoff_col] >= color_cutoff

        def get_category(row):
            if row[color_cutoff_col] >= color_cutoff:
                if row["Num_Papers"]==0:
                    return f"{organ}, {metric}: above cutoff and new"
                else:
                    return f"{organ}, {metric}: above cutoff and known"
            else:
                return f"{organ}, {metric}: below cutoff"

        merged_df_this_organ["category"] = merged_df_this_organ.apply(get_category, axis=1)

        sns.stripplot(data=merged_df_this_organ, x="bin", y="Num_Papers_Log1p", ax=plt.gca(), jitter=True, hue="category", size=1, dodge=False, 
                        palette={f"{organ}, {metric}:above cutoff and new":"red",f"{organ}, {metric}:above cutoff and known":"black", f"{organ}, {metric}:below cutoff":"grey"})
        sns.violinplot(data=merged_df_this_organ, x="bin", y="Num_Papers_Log1p", ax=plt.gca(),inner=None, hue="above_color_cutoff", palette={True:"black",False:"grey"}, alpha=0.2, cut=0,dodge=False)                                                                                                                            
        # Add median lines
        for i, bin in enumerate(merged_df_this_organ["bin"].cat.categories):
            median = merged_df_this_organ[merged_df_this_organ["bin"]==bin]["Num_Papers_Log1p"].median()
            plt.plot([i-0.2, i+0.2], [median, median], color="black", linewidth=2)

            # write N numbers underneith
            N = merged_df_this_organ[merged_df_this_organ["bin"]==bin].shape[0]
            plt.text(i, -0.8, f"N total={N}", fontsize=6, ha='center', va='bottom')

            no_ref_df= merged_df_this_organ[(merged_df_this_organ["bin"]==bin) & (merged_df_this_organ["Num_Papers"]==0)]
            n_no_ref = no_ref_df.shape[0]
            if merged_df_this_organ[(merged_df_this_organ["bin"]==bin) & (merged_df_this_organ["above_color_cutoff"]==True)].shape[0] > 0:
                plt.text(i, -0.4, f"N new={n_no_ref}", fontsize=6, ha='center', va='bottom', color="red")
                #print(f"New hits for bin {bin}:\n {no_ref_df[['names', metric, 'pvals_adj']]}")

            # test for significance vs the first bin
            ref_bin=merged_df_this_organ["bin"].cat.categories[0]
            print(f"Testing significance for bin {bin} vs {ref_bin}")
            if bin != ref_bin:
                ref_papers_log1p= merged_df_this_organ[merged_df_this_organ["bin"]==ref_bin]["Num_Papers_Log1p"].values
                test_papers_log1p= merged_df_this_organ[merged_df_this_organ["bin"]==bin]["Num_Papers_Log1p"].values

                if ref_papers_log1p.size == 0 or test_papers_log1p.size == 0:
                    print(f"Skipping significance test for bin {bin} vs {ref_bin}: one of the bins is empty")
                    continue
                # test for significance (Mann-Whitney U test)
                u_stat, pval = stats.mannwhitneyu(ref_papers_log1p, test_papers_log1p, alternative="two-sided")
                stars= ""
                if pval < 0.001:
                    stars = "***"
                elif pval < 0.01:
                    stars = "**"
                elif pval < 0.05:
                    stars = "*"
                # plt.text(i, merged_df_this_organ["Num_Papers_Log1p"].max()*1.01, stars, fontsize=6, ha='center', va='bottom', color="black")
                # plot nicely formated pval instead
                plt.text(i, merged_df_this_organ["Num_Papers_Log1p"].max()*1.01, f"p={pval:.2e}", fontsize=6, ha='center', va='bottom', color="black")


        rect = patches.Rectangle(start_box, width_box, 1, linewidth=1, edgecolor='red', facecolor="none")
        plt.gca().add_patch(rect)
        plt.xlabel(f"Enrichment Strength ({metric})")
        plt.ylabel("Log10(Number of PubMed IDs + 1)")
        plt.title(f"Literature Support vs Enrichment Strength\nfor {organ}")
        plt.xticks(rotation=45,ha='right')
        
        # remove legend
        plt.legend([],[], frameon=False)

        merged_dfs_per_organ[organ] = merged_df_this_organ


    plt.tight_layout()
    plt.savefig(outdir / f"enrichment_vs_literature_all_organs.pdf", dpi=300)
    plt.show()

    return merged_dfs_per_organ


        # # start a new figure and plot here
        # old_fig = plt.gcf()
        # new_fig = plt.figure(figsize=(8, 6))
        # # hex joint plot
        # ax = new_fig.gca()
        # ax.hexbin(
        #     x=merged_df_this_organ[enrichment_strength_col],
        #     y=merged_df_this_organ["Num_Papers_Log1p"],
        #     gridsize=50,
        #     cmap="Greys",
        #     bins="log",
        # )
        # sns.regplot(x=merged_df_this_organ[enrichment_strength_col], y=merged_df_this_organ["Num_Papers_Log1p"], scatter=False,ax=new_fig.gca(),line_kws={"color":"black"}, lowess=lowess)
        # sns.scatterplot(data=merged_df_this_organ[merged_df_this_organ[color_cutoff_col] >= color_cutoff], x=enrichment_strength_col,
        #                 y="Num_Papers_Log1p", alpha=0.3,ax=new_fig.gca(),size=0.2)

        # new_fig.show()

        # # Now plot a the average Num_Papers_Log1p values over different bins of enrichment strength
        # new_fig = plt.figure(figsize=(8, 6))    
        # ax = new_fig.gca()
        # plt.sca(ax)

        # # binning
        # bins = np.linspace(0, merged_df_this_organ[enrichment_strength_col].max(), 10)
        # bin_means = []
        # bin_edges = []
        # bin_stds = []
        # bin_counts = []
        # for i in range(len(bins)-1):
        #     bin_mask = (merged_df_this_organ[enrichment_strength_col] >= bins[i]) & (merged_df_this_organ[enrichment_strength_col] < bins[i+1])
        #     bin_means.append(merged_df_this_organ[bin_mask]["Num_Papers_Log1p"].mean())
        #     bin_stds.append(merged_df_this_organ[bin_mask]["Num_Papers_Log1p"].std())
        #     bin_counts.append(bin_mask.sum())
        #     bin_edges.append((bins[i] + bins[i+1]) / 2)
        # bin_means = np.array(bin_means)
        # bin_stds = np.array(bin_stds)
        # bin_counts = np.array(bin_counts)
        # bin_edges = np.array(bin_edges)
        # # plot
        # plt.errorbar(bin_edges, bin_means, yerr=bin_stds, fmt='o', color='black', capsize=5)
        # plt.xlabel(f"Enrichment Strength ({enrichment_strength_col})")
        # plt.ylabel("Mean Log10(Number of PubMed IDs + 1)")
        # plt.title(f"Mean Literature Support vs Enrichment Strength\nfor {organ}")
        # plt.show()

        # new_fig = plt.figure(figsize=(8, 6))    
        # ax = new_fig.gca()
        # plt.sca(ax)
        # # assign a bin to each gene

        # # plot
        # sns.boxplot(data=merged_df_this_organ, x="bin", y="Num_Papers_Log1p", ax=ax)
        # plt.xlabel(f"Enrichment Strength ({enrichment_strength_col})")
        # plt.ylabel("Log10(Number of PubMed IDs + 1)")
        # plt.title(f"Literature Support vs Enrichment Strength\nfor {organ}")
        # plt.xticks(rotation=45)
        # plt.show()

        # new_fig = plt.figure(figsize=(8, 6))    
        # ax = new_fig.gca()
        # plt.sca(ax)

        # # plot
        # sns.stripplot(data=merged_df_this_organ, x="bin", y="Num_Papers_Log1p", ax=ax, jitter=True, hue="above_color_cutoff", palette={True:"black",False:"grey"},size=0.5, dodge=False)
        # sns.violinplot(data=merged_df_this_organ, x="bin", y="Num_Papers_Log1p", ax=ax,inner=None, hue="above_color_cutoff", palette={True:"black",False:"grey"}, alpha=0.2, cut=True,dodge=False)
        # # Add median lines
        # for i, bin in enumerate(merged_df_this_organ["bin"].cat.categories):
        #     median = merged_df_this_organ[merged_df_this_organ["bin"]==bin]["Num_Papers_Log1p"].median()
        #     plt.plot([i-0.4, i], [median, median], color="black", linewidth=1)
        # plt.xlabel(f"Enrichment Strength ({enrichment_strength_col})")
        # plt.ylabel("Log10(Number of PubMed IDs + 1)")
        # plt.title(f"Literature Support vs Enrichment Strength\nfor {organ}")
        # plt.xticks(rotation=45)
        # plt.show()

        # new_fig = plt.figure(figsize=(8, 6))    
        # ax = new_fig.gca()
        # plt.sca(ax)







        
def classify_dominant_organ(adata, marker_gene_dict, score_prefix, quantile_thresh=0.95):
    """Classifies cells based on highest mean expression of organ marker genes."""
    organ_expression = {}
    valid_organs = []

    print(f"Calculating mean expression for {score_prefix} markers...")
    for organ, genelist in marker_gene_dict.items():
        valid_genes = [gene for gene in genelist if gene in adata.var_names]
        if valid_genes:
            # POTENTIAL ISSUE: Mean of raw counts might not be ideal. Consider using normalized data.
            # Using adata.X assumes processed (e.g., log-normalized) data.
            mean_expr = adata[:, valid_genes].X.mean(axis=1)
            # Ensure result is 1D array
            organ_expression[organ] = np.array(mean_expr).flatten()
            valid_organs.append(organ)
        else:
            print(f"Skipping {organ} for dominant classification: No valid genes.")
            pass # Skip organ if no genes

    if not organ_expression:
        warnings.warn("Cannot classify dominant organ: No valid marker genes found for any organ.")
        return pd.Series(index=adata.obs.index, dtype=str)

    expression_df = pd.DataFrame(organ_expression, index=adata.obs.index)

    # Compute threshold: top % for each organ
    thresholds = expression_df.quantile(quantile_thresh, axis=0)

    # Identify the dominant organ per cell
    dominant_organ_series = expression_df.idxmax(axis=1)

    # Map each cell’s dominant organ to its corresponding threshold
    dominant_thresholds = dominant_organ_series.map(thresholds)

    # Filter out cells where max expression does not exceed the specific organ's threshold
    max_expression_per_cell = expression_df.max(axis=1)
    below_threshold_mask = max_expression_per_cell < dominant_thresholds
    dominant_organ_series[below_threshold_mask] = "None" # Assign "None"

    print(f"Classified dominant organ using {score_prefix} markers.")
    return dominant_organ_series

# --- Perform Enrichment Tests ---
def run_disease_enrichment(marker_gene_dict, disease_gene_dict, background_size):
    """Runs Fisher's exact test for disease enrichment in marker sets."""
    enrichment_results = {}
    pvals = []
    keys = []
    overlaps = defaultdict(list)

    for disease_name, disease_genes in disease_gene_dict.items():
        if not disease_genes: continue # Skip if no genes for this disease
        for organ, marker_genes in marker_gene_dict.items():
            if not marker_genes: continue # Skip if no marker genes for this organ

            # Ensure sets contain strings
            set_disease = set(map(str, disease_genes))
            set_marker = set(map(str, marker_genes))

            pval = fishers_exact_test(set_disease, set_marker, background_size)
            enrichment_results[(disease_name, organ)] = pval
            pvals.append(pval)
            keys.append((disease_name, organ))

            this_overlap= set_disease & set_marker
            for overlap_gene in this_overlap:
                overlaps[organ].append((overlap_gene, disease_name))
    # FDR correction
    if pvals:
         fdrs = adjust_fdr_bh(pvals)
         for i, key in enumerate(keys):
             enrichment_results[key] = {"pval": pvals[i], "fdr": fdrs[i]}
    else: # Add FDR column even if empty
        for key in enrichment_results:
             enrichment_results[key] = {"pval": enrichment_results[key], "fdr": np.nan}

    # turn overlaps into a dataframe
    for organ in overlaps.keys():
        overlaps[organ] = pd.DataFrame(overlaps[organ], columns=["Gene", "Disease"])
        # set Gene as index
        overlaps[organ].set_index("Gene", inplace=True)


    return pd.DataFrame.from_dict(enrichment_results, orient='index'), overlaps

# --- Bipartite Graph: Dominant Organ vs Original Annotation ---
def plot_bipartite_composition(adata, dominant_organ_col, anno_col, title, filename, outdir):
    """Plots a bipartite graph showing composition of dominant organ groups."""
    if dominant_organ_col not in adata.obs.columns or anno_col not in adata.obs.columns:
        warnings.warn(f"Skipping bipartite plot: Columns {dominant_organ_col} or {anno_col} missing.")
        return

    # Filter out "None" dominant organ assignments
    valid_mask = adata.obs[dominant_organ_col] != "None"
    if not valid_mask.any():
        print("Skipping bipartite plot: No cells with valid dominant organ assignments.")
        return

    filtered_obs = adata.obs.loc[valid_mask, [dominant_organ_col, anno_col]]

    G = nx.Graph()
    organs = sorted(filtered_obs[dominant_organ_col].unique())
    annotations = sorted(filtered_obs[anno_col].unique())

    # Add nodes with bipartite attribute
    G.add_nodes_from(organs, bipartite=0) # Layer 0: Dominant Organs
    G.add_nodes_from(annotations, bipartite=1) # Layer 1: Annotations

    # Calculate edge weights (counts) and node sizes
    edge_weights = defaultdict(int)
    anno_counts = Counter(filtered_obs[anno_col])
    organ_counts = Counter(filtered_obs[dominant_organ_col])

    for _, row in filtered_obs.iterrows():
        edge_weights[(row[dominant_organ_col], row[anno_col])] += 1

    # Add edges with weights
    for (organ, anno), weight in edge_weights.items():
        if organ in G and anno in G: # Ensure nodes exist
            G.add_edge(organ, anno, weight=weight)

    # --- Prepare for plotting ---
    pos = {}
    organ_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    anno_nodes = set(G) - organ_nodes

    # Position nodes in two columns
    y_step_organ = 1.0 / (len(organ_nodes) + 1) if len(organ_nodes) > 0 else 1
    for i, node in enumerate(sorted(list(organ_nodes))):
        pos[node] = (-1, 1 - (i + 1) * y_step_organ) # Left column

    y_step_anno = 1.0 / (len(anno_nodes) + 1) if len(anno_nodes) > 0 else 1
    for i, node in enumerate(sorted(list(anno_nodes))):
        pos[node] = (1, 1 - (i + 1) * y_step_anno) # Right column

    # Node sizes (proportional to counts) and colors
    node_sizes = []
    node_colors = []
    base_node_size = 50
    max_organ_count = max(organ_counts.values()) if organ_counts else 1
    max_anno_count = max(anno_counts.values()) if anno_counts else 1

    for node in G.nodes():
        if node in organ_nodes:
            node_colors.append(ORGAN_COLORS.get(node, 'gray'))
            # Scale size, ensure minimum size
            size = base_node_size + 500 * (organ_counts.get(node, 0) / max_organ_count)
            node_sizes.append(max(size, base_node_size))
        else: # Annotation nodes
            node_colors.append('lightcoral')
            size = base_node_size + 500 * (anno_counts.get(node, 0) / max_anno_count)
            node_sizes.append(max(size, base_node_size))


    # Edge widths (proportional to weight)
    edge_widths = [G[u][v]['weight'] * 0.1 for u, v in G.edges()] # Scale weights for width

    plt.figure(figsize=(12, max(8, 0.3 * len(anno_nodes)))) # Adjust height based on annotations
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors,
            width=edge_widths, edge_color="gray", alpha=0.7, font_size=8)
    plt.title(title)
    filepath = outdir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved bipartite plot to: {filepath}")
    plt.show()
    plt.close()


def get_highest_expression_day(adata):
    """
    Calculate mean expression of each gene per day and identify the day 
    with highest expression for each gene.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with day information in adata.obs['day']
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with gene names and their highest expression day
    """
    # Get unique days
    days = adata.obs["anno_og_time_days"].unique()
    
    # Initialize dictionary to store mean expression per day
    mean_expr_by_day = {}
    q97_expr_by_day ={}
    
    # Calculate mean expression for each day
    for day in days:
        # Get cells for this day
        day_mask = adata.obs['anno_og_time_days'] == day
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