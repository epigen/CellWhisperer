import scib
from anndata import AnnData
from typing import Dict
import numpy as np
import logging
import scanpy as sc
import json
import logging
import re
from cellwhisperer.config import get_path
import warnings

TABSAP_WELLSTUDIED_COLORMAPPING = {
                'erythrocyte': (0.5, 0, 0), 
                'macrophage': 'darkblue',
                'monocyte': 'lightblue',
                'non-classical monocyte': 'teal',
                'classical monocyte': 'steelblue',
                'neutrophil': 'red',
                'basophil': 'grey',
                'nk cell': 'orange',
                'cd8-positive, alpha-beta t cell': 'darkgreen',
                'cd4-positive, alpha-beta memory t cell': 'olive',
                'cd4-positive, alpha-beta t cell': 'lightgreen',
                'cd8-positive, alpha-beta cytokine secreting effector t cell': 'forestgreen',
                'naive b cell': 'violet',
                'memory b cell': 'darkviolet',
                'capillary endothelial cell': 'pink',
                'type ii pneumocyte': 'tan',
                'respiratory goblet cell': 'sandybrown',
                'club cell': 'aquamarine',
                'basal cell': 'lightgray',
                'hepatocyte': 'goldenrod',
            }
PANCREAS_ORDER = ["alpha","beta","gamma","delta","epsilon",
                           "acinar","ductal",
                           "endothelial",
                           "activated stellate","quiescent stellate", "schwann",
                           "macrophage", "mast","t cell"]

# FROM: https://github.com/microsoft/zero-shot-scfoundation/blob/f0353a49cb6aff5c8105cc6ddd755efa8fbab98d/sc_foundation_evals/utils.py#L16
# MODIFIED wrapper for all scib metrics from 
# https://github.com/bowang-lab/scGPT/blob/5a69912232e214cda1998f78e5b4a7b5ef09fe06/scgpt/utils/util.py#L267
def eval_scib_metrics(
    adata: AnnData,
    batch_key: str = "str_batch",
    label_key: str = "cell_type",
    embedding_key: str = "X_scGPT"
) -> Dict:
    
    # if adata.uns["neighbors"] exists, remove it to make sure the optimal 
    # clustering is calculated for the correct embedding
    # print a warning for the user
    if "neighbors" in adata.uns:        
        logging.warning(f"neighbors in adata.uns found \n {adata.uns['neighbors']} "
                    "\nto make sure the optimal clustering is calculated for the "
                    "correct embedding, removing neighbors from adata.uns."
                    "\nOverwriting calculation of neighbors with "
                    f"sc.pp.neighbors(adata, use_rep={embedding_key}).")
        adata.uns.pop("neighbors", None)
        sc.pp.neighbors(adata, use_rep=embedding_key)
        logging.info("neighbors in adata.uns removed, new neighbors calculated: "
                 f"{adata.uns['neighbors']}")


    # in case just one batch scib.metrics.metrics doesn't work 
    # call them separately
    results_dict = dict()

    res_max, nmi_max, nmi_all = scib.metrics.clustering.opt_louvain(
            adata,
            label_key=label_key,
            cluster_key="cluster",
            use_rep=embedding_key,
            function=scib.metrics.nmi,
            plot=False,
            verbose=False,
            inplace=True,
            force=True,
    )
    
    results_dict["NMI_cluster/label"] = scib.metrics.nmi(
        adata, 
        "cluster",
        label_key,
        "arithmetic",
        nmi_dir=None
    )

    results_dict["ARI_cluster/label"] = scib.metrics.ari(
        adata, 
        "cluster", 
        label_key
    )

    results_dict["ASW_label"] = scib.metrics.silhouette(
        adata, 
        label_key, 
        embedding_key, 
        "euclidean"
    )   

    results_dict["graph_conn"] = scib.metrics.graph_connectivity(
        adata,
        label_key=label_key
    )
    

    # Calculate this only if there are multiple batches
    if len(adata.obs[batch_key].unique()) > 1:
        results_dict["ASW_batch"] = scib.metrics.silhouette(
            adata,
            batch_key,
            embedding_key,
            "euclidean"
        )

        results_dict["ASW_label/batch"] = scib.metrics.silhouette_batch(
            adata, 
            batch_key,
            label_key, 
            embed=embedding_key, 
            metric="euclidean",
            return_all=False,
            verbose=False
        )

        results_dict["PCR_batch"] = scib.metrics.pcr(
            adata,
            covariate=batch_key,
            embed=embedding_key,
            recompute_pca=True,
            n_comps=50,
            verbose=False
        )

    results_dict["avg_bio"] = np.mean(
        [
            results_dict["NMI_cluster/label"],
            results_dict["ARI_cluster/label"],
            results_dict["ASW_label"],
        ]
    )

    logging.debug(
        "\n".join([f"{k}: {v:.4f}" for k, v in results_dict.items()])
    )

    # remove nan value in result_dict
    results_dict = {k: v for k, v in results_dict.items() if not np.isnan(v)}

    # remove "neighbor" from adata.uns
    if "neighbors" in adata.uns:
        adata.uns.pop("neighbors", None)

    return results_dict


def umap_on_embedding(adata,
                      embedding_key:str="X_cellwhisperer",
                      neighbors_key:str="neighbors_cellwhisperer",
                      umap_key:str="X_umap_on_neighbors_cellwhisperer") -> AnnData:
    """
    Calculate UMAP on the given embedding and store it in adata.obsm[umap_key].
    :param adata: anndata.AnnData instance.
    :param embedding_key: Key in adata.obsm where the embedding is stored.
    :param neighbors_key: Key in adata.uns where the neighbors should be stored.
    :param umap_key: Key in adata.obsm where the UMAP embedding should be stored.
    :return: adata with UMAP embedding stored in adata.obsm[umap_key].
    """
    sc.pp.neighbors(adata, use_rep=embedding_key,key_added=neighbors_key)
    adata.obsm[umap_key]=sc.tl.umap(adata, neighbors_key=neighbors_key, min_dist = 0.3,copy=True).obsm["X_umap"]
    return adata


def count_keywords_in_dataset(keywords,dataset="archs4_metasra"):
    """
    Count the number of samples in the dataset in which each keyword appears at least once.
    :param dataset: the dataset to count the keywords in.
    :param keywords: the keywords to count.
    
    :return: a dictionary with the keywords as keys and the number of samples in which they appear as values.
    """

    with open(get_path(["paths","structured_annotations"],dataset=dataset)) as f:
        structured_annotations = json.load(f)

    keywords_counts = {term:0 for term in keywords}
    for keyword in keywords_counts.keys():

        # generate all replacements:
        keywords_and_alternatives = set([keyword.lower()])
        for replace_from_char in ["-", "_"," "]:
            for replace_to_char in ["-", "_"," "]:
                if replace_from_char != replace_to_char:
                    keywords_and_alternatives.add(keyword.lower().replace(replace_from_char, replace_to_char))

        for sample_id, sample_dict in structured_annotations.items():
            for value in sample_dict.values():
                if value is not None:
                    # look if any of the keywords is in the value:
                    found = re.findall("|".join(keywords_and_alternatives), value.lower())
                    if len(found) > 0:
                        keywords_counts[keyword] += 1
                        break

    return keywords_counts