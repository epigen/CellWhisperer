from anndata import AnnData
import scanpy as sc
import pandas as pd

## Define the suffix and prefix for the text embeddings
SUFFIX_PREFIX_DICT = {}
SUFFIX_PREFIX_DICT["celltype"] = ("A sample of ", " from a healthy individual")
SUFFIX_PREFIX_DICT["organ_tissue"] = ("A sample of ", " from a healthy individual")
SUFFIX_PREFIX_DICT["Disease"] = ("A sample from an individual with ","")
SUFFIX_PREFIX_DICT["Disease_subtype"] = ("A sample from an individual with ","")
SUFFIX_PREFIX_DICT["Tissue"] = ("A "," sample")
SUFFIX_PREFIX_DICT["Tissue_subtype"] = ("A "," sample")

TABSAP_WELLSTUDIED_COLORMAPPING = {
                'erythrocyte': (0.5, 0, 0), 
                'macrophage': 'darkblue',
                'monocyte': 'lightblue',
                'non-classical monocyte': 'teal',
                'classical monocyte': 'steelblue',
                'neutrophil': 'red',
                'basophil': 'black',
                'NK cell': 'orange',
                'CD8-positive, alpha-beta T cell': 'darkgreen',
                'CD4-positive, alpha-beta memory T cell': 'olive',
                'CD4-positive, alpha-beta T cell': 'lightgreen',
                'CD8-positive, alpha-beta cytokine secreting effector T cell': 'forestgreen',
                'naive B cell': 'violet',
                'memory B cell': 'darkviolet',
                'capillary endothelial cell': 'pink',
                'type II pneumocyte': 'tan',
                'respiratory goblet cell': 'sandybrown',
                'club cell': 'aquamarine',
                'basal cell': 'tomato',
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
    adata.obsm[umap_key]=sc.tl.umap(adata, neighbors_key=neighbors_key, copy=True).obsm["X_umap"]
    return adata


def prepare_integration_df(result_metrics_dict:dict) -> pd.DataFrame:
    """
    Restructure the result_metrics_dict to a pandas DataFrame for easier plotting.
    :param result_metrics_dict: Dictionary containing the integration scores.
    :return: pandas DataFrame containing the integration scores.
    """
    integration_scores_df = pd.DataFrame(result_metrics_dict).T
    integration_scores_df = integration_scores_df.reset_index()
    integration_scores_df=integration_scores_df.rename(columns={"level_0":"dataset_name",
                                                                "level_1":"Method"})
    integration_scores_df["Method"]=integration_scores_df["Method"].str.capitalize()
    integration_scores_df=integration_scores_df.rename(columns={"ASW_label__batch":"Batch\nintegration\nscore",
                                                                "avg_bio":"Cell type\nintegration score\n(avg)",
                                                                "ASW_label":"Cell type\nintegration score\n(ASW)"})
    integration_scores_df=integration_scores_df[["dataset_name","Method","Batch\nintegration\nscore","Cell type\nintegration score\n(avg)", "Cell type\nintegration score\n(ASW)"]]
    integration_scores_df=pd.melt(integration_scores_df,id_vars=["dataset_name","Method"],value_vars=["Batch\nintegration\nscore","Cell type\nintegration score\n(avg)", "Cell type\nintegration score\n(ASW)"],var_name="metric",value_name="value")
    return integration_scores_df



