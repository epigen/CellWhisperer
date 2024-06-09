import scanpy as sc
import anndata
from cellwhisperer.config import get_path
import numpy as np
import pandas as pd
from zero_shot_validation_scripts.utils import TABSAP_WELLSTUDIED_COLORMAPPING


def load_dataset(read_count_table_path: str,
                 processed_data_path: str,
                    transcriptome_model_name: str) -> anndata.AnnData:
    """Load the dataset's adata, cellwhisperer embeddings and transcriptome features"""
    
    adata = anndata.read_h5ad(
        read_count_table_path
    )
    processed_data = np.load(processed_data_path, allow_pickle=True)

    assert (processed_data["orig_ids"] == adata.obs.index).all()

    adata.obsm["X_cellwhisperer"] = processed_data["transcriptome_embeds"]
    adata.obsm[f"X_{transcriptome_model_name}"] = processed_data["transcriptome_features"]

    return adata




def preprocess_immgen(adata: anndata.AnnData) -> anndata.AnnData:
    """Preprocess the immgen dataset."""

    translation_dict={'B':"B cells",
                     'DC':'Dendritic cells',
                     'ILC':'Natural Killer cells',
                     'Mo':'Monocytes',
                     'T':'T cells'}

    adata.obs["celltype"] = [
        translation_dict[x.split(".")[0]] for x in adata.obs_names #["natural_language_annotation"]
    ]
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    adata.obs["batch"] = "1"

    return adata.copy()


def preprocess_tabula_sapiens(
    adata: anndata.AnnData, well_studied_only=False, min_100=False
) -> anndata.AnnData:
    """Preprocess the tabula_sapiens_100_cells_per_type or the full tabula_sapiens dataset."""
    adata.obs["celltype"] = adata.obs["cell_ontology_class"]

    # replace the following labels with properly capitalized ones:

    # format: old substring, new substring, old full strings
    capitalization_list = [
        ("cd", "CD", ['cd8-positive, alpha-beta t cell', 'cd4-positive, alpha-beta t cell', 'cd4-positive, alpha-beta memory t cell',
                       'cd8-positive, alpha-beta cytokine secreting effector t cell', 'cd141-positive myeloid dendritic cell',
                         'naive thymus-derived cd4-positive, alpha-beta t cell', 'cd8-positive alpha-beta t cell',
                           'cd4-positive alpha-beta t cell', 'cd1c-positive myeloid dendritic cell', 'cd4-positive helper t cell',
                             'cd8-positive, alpha-beta memory t cell', 'naive thymus-derived cd8-positive, alpha-beta t cell',
                               'cd8-positive, alpha-beta cytotoxic t cell', 'cd24 neutrophil', 'cd8b-positive nk t cell']),
        ("nk t cell","NKT cell", ['type i nk t cell', 'CD8b-positive nk t cell', 'mature nk t cell']),
        ("nkt cell","NKT cell",['nkt cell']),
        ("nk cell","NK cell", ["nk cell"]),
        ("t cell","T cell", ['t cell', 'CD8-positive, alpha-beta t cell', 'CD4-positive, alpha-beta t cell',
                              'CD4-positive, alpha-beta memory t cell', 'CD8-positive, alpha-beta cytokine secreting effector t cell',
                              'naive thymus-derived CD4-positive, alpha-beta t cell', 'CD8-positive alpha-beta t cell', 'CD4-positive alpha-beta t cell', 
                              'regulatory t cell', 'CD4-positive helper t cell', 'CD8-positive, alpha-beta memory t cell', 
                              'naive thymus-derived CD8-positive, alpha-beta t cell', 'CD8-positive, alpha-beta cytotoxic t cell',
                              'naive regulatory t cell', 'dn1 thymic pro-t cell']),
        ("b cell", "B cell", ['b cell', 'naive b cell', 'memory b cell']),
        ("dn4","DN4",['dn4 thymocyte']),
        ("dn3","DN3",['dn3 thymocyte']),
        ("dn1","DN1",['dn1 thymic pro-T cell']),
        ("type ii","type II",['type ii pneumocyte']),
        ("type i","type I",['type i NKT cell', 'type i pneumocyte']),
        ('pancreatic pp cell','Pancreatic PP cell',['pancreatic pp cell']),
    ]
    for old_substring, new_substring, old_full_strings in capitalization_list:
        adata.obs["celltype"] = [x.replace(old_substring, new_substring) if x in old_full_strings else x for x in adata.obs["celltype"]]

    adata.obs["celltype"] = adata.obs["celltype"].astype("category")

    adata.obs["batch"] = (
        adata.obs["donor"].astype(str) + "_" + adata.obs["method"].astype(str)
    )
    if "raw_counts" in adata.layers.keys():
        adata.X = adata.layers["raw_counts"].copy()
        #raise ValueError("Raw counts should be in adata.X, not in adata.layers['raw_counts']")
    if well_studied_only:
        adata = adata[
            adata.obs["celltype"].isin(list(TABSAP_WELLSTUDIED_COLORMAPPING.keys())), :
        ]
    if min_100:
        value_counts_per_celltype = adata.obs["celltype"].value_counts()
        celltypes_to_keep = value_counts_per_celltype[
            value_counts_per_celltype >= 100
        ].index.values
        adata = adata[adata.obs["celltype"].isin(celltypes_to_keep), :]

    adata.obsm["X_umap_original"] = adata.obsm["X_umap"]

    return adata.copy()


def preprocess_pancreas(adata: anndata.AnnData) -> anndata.AnnData:
    """Preprocess the pancreas dataset."""
    # NOTE data does not actually contain raw counts.
    adata.obs["batch"] = adata.obs["tech"]
    adata.X = adata.layers["counts"]
    adata.var["gene_name"] = adata.var.index
    adata.obs["celltype"] = [x.replace("_", " ") for x in adata.obs["celltype"]]
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    return adata.copy()

def preprocess_human_disease(adata: anndata.AnnData) -> anndata.AnnData:
    """Preprocess the human_disease dataset."""
    
    adata.obs["celltype"] = adata.obs["Disease_subtype"]
    adata.obs["celltype"] = adata.obs["celltype"].astype("str").astype("category")
    adata.obs[
        "batch"
    ] = "1"  # NOTE Could use "sra_study_acc" instead of this dummy - this will likely be strongly correlated with the celltype though..
    adata.obs["Treated"] = adata.obs["Treated"].astype("bool")
    adata.obs["Treated"] = [
        "treatment status: treated" if x else "treatment status: untreated"
        for x in adata.obs["Treated"]
    ]
    adata.obs["Treated"] = adata.obs["Treated"].astype("category")
    return adata.copy()


def load_and_preprocess_dataset(dataset_name: str,
                                read_count_table_path: str,
                 processed_data_path: str,
                    transcriptome_model_name: str) -> anndata.AnnData:
    """Preprocess the dataset based on the provided dataset name and paths."""


    adata = load_dataset(read_count_table_path = read_count_table_path,
                            processed_data_path = processed_data_path,
                            transcriptome_model_name = transcriptome_model_name
    )

    if "tabula_sapiens" in dataset_name:
        adata = preprocess_tabula_sapiens(adata)
        well_studied_only = "well_studied_celltypes" in dataset_name
        min_100 = "min_100" in dataset_name
        adata = preprocess_tabula_sapiens(
            adata, well_studied_only=well_studied_only, min_100=min_100
        )
    elif dataset_name == "pancreas":
        adata = preprocess_pancreas(adata)
    elif dataset_name == "human_disease":
        adata = preprocess_human_disease(adata)
    elif dataset_name == "immgen":
        adata = preprocess_immgen(adata)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Very basic QC (as in zero-shot paper)
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=10)

    # Just to be sure and to avoid warnings later, remove the original neighbors/umaps/pcas
    if "neighbors" in adata.uns.keys():
        del adata.uns["neighbors"]
    if "X_umap" in adata.obsm.keys():
        del adata.obsm["X_umap"]
    if "X_pca" in adata.obsm.keys():
        del adata.obsm["X_pca"]

    return adata.copy()
