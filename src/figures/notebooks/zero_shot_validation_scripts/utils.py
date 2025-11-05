import pandas as pd

## Define the suffix and prefix for the text embeddings
SUFFIX_PREFIX_DICT = {}
SUFFIX_PREFIX_DICT["celltype"] = ("A sample of ", " from a healthy individual")
SUFFIX_PREFIX_DICT["organ_tissue"] = ("A sample of ", " from a healthy individual")
SUFFIX_PREFIX_DICT["Disease"] = ("A sample from an individual with ", "")
SUFFIX_PREFIX_DICT["Disease_subtype"] = ("A sample from an individual with ", "")
SUFFIX_PREFIX_DICT["Tissue"] = ("A ", " sample")
SUFFIX_PREFIX_DICT["Tissue_subtype"] = ("A ", " sample")


TABSAP_WELLSTUDIED_COLORMAPPING = {
    "erythrocyte": (0.5, 0, 0),
    "macrophage": "darkblue",
    "monocyte": "lightblue",
    "non-classical monocyte": "teal",
    "classical monocyte": "steelblue",
    "neutrophil": "red",
    "basophil": "black",
    "NK cell": "orange",
    "CD8-positive, alpha-beta T cell": "darkgreen",
    "CD4-positive, alpha-beta memory T cell": "olive",
    "CD4-positive, alpha-beta T cell": "lightgreen",
    "CD8-positive, alpha-beta cytokine secreting effector T cell": "forestgreen",
    "naive B cell": "violet",
    "memory B cell": "darkviolet",
    "capillary endothelial cell": "pink",
    "type II pneumocyte": "tan",
    "respiratory goblet cell": "sandybrown",
    "club cell": "aquamarine",
    "basal cell": "tomato",
    "hepatocyte": "goldenrod",
}
PANCREAS_ORDER=[
    'alpha cell',
    'beta cell',
    'gamma cell',
    'delta cell',
    'epsilon cell',
    'acinar cell',
    'ductal cell',
    'endothelial cell',
    'activated stellate cell',
    'quiescent stellate cell',
    'schwann cell',
    'macrophage',
    'mast cell',
    'T cell'
]


def umap_on_embedding(
    adata,
    embedding_key: str = "X_cellwhisperer",
    neighbors_key: str = "neighbors_cellwhisperer",
    umap_key: str = "X_umap_on_neighbors_cellwhisperer",
):
    """
    Calculate UMAP on the given embedding and store it in adata.obsm[umap_key].
    :param adata: anndata.AnnData instance.
    :param embedding_key: Key in adata.obsm where the embedding is stored.
    :param neighbors_key: Key in adata.uns where the neighbors should be stored.
    :param umap_key: Key in adata.obsm where the UMAP embedding should be stored.
    :return: adata with UMAP embedding stored in adata.obsm[umap_key].
    """
    import scanpy as sc

    sc.pp.neighbors(adata, use_rep=embedding_key, key_added=neighbors_key)
    adata.obsm[umap_key] = sc.tl.umap(
        adata, neighbors_key=neighbors_key, copy=True
    ).obsm["X_umap"]
    return adata
