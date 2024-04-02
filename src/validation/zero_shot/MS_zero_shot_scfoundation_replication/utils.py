from anndata import AnnData
from typing import Dict
import scanpy as sc
import json
import re
from cellwhisperer.config import get_path

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