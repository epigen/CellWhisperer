import scanpy as sc
import anndata
from cellwhisperer.config import get_path
import numpy as np
import pandas as pd
from zero_shot_validation_scripts.utils import TABSAP_WELLSTUDIED_COLORMAPPING
from typing import Optional, Mapping, Tuple


def preprocess_aida(adata: anndata.AnnData, level=1) -> anndata.AnnData:
    """
    Translate cell type names from abbreviations to proper names (based on GPT4o)
    """

    adata.obs["full_cell_type_name"] = adata.obs["cell_type"].copy()


    if level == 1:
        cell_type_dict = {
            'B': 'B cell',
            'CD34_HSPC': 'CD34-positive hematopoietic stem and progenitor cell',
            'DC': 'dendritic cell',
            'ILC': 'innate lymphoid cell',
            'Myeloid': 'myeloid cell',
            'NK': 'natural killer cell',
            'Plasma_Cell': 'plasma cell',
            'Platelet': 'platelet',
            'T': 'T cell',
        }
        level_name = "Annotation_Level1"
    elif level == 2:
        cell_type_dict = {
            'B': 'B cell',
            'CD16+_NK': 'CD16-positive natural killer cell',
            'CD34_HSPC': 'CD34-positive hematopoietic stem and progenitor cell',
            'CD4+_T': 'CD4-positive T cell',
            'CD56+_NK': 'CD56-positive natural killer cell',
            'CD8+_T': 'CD8-positive T cell',
            'DC': 'dendritic cell',
            'ILC': 'innate lymphoid cell',
            'Monocyte': 'monocyte',
            'Myeloid': 'myeloid cell',
            'NK': 'natural killer cell',
            'Plasma_Cell': 'plasma cell',
            'Platelet': 'platelet',
            'T': 'T cell',
            'atypical_B': 'atypical B cell',
            'cDC': 'conventional dendritic cell',
            'dnT': 'double-negative T cell',
            'qdT': 'quadruple-negative T cell',
            'memory_B': 'memory B cell',
            'naive_B': 'naive B cell',
            'pDC': 'plasmacytoid dendritic cell',
        }
        level_name = "Annotation_Level2"
    else:
        raise ValueError("level must be 1 or 2")


    adata.obs["celltype"] = adata.obs[level_name].map(cell_type_dict)

    return adata.copy()


def preprocess_immgen(adata: anndata.AnnData) -> anndata.AnnData:
    """Preprocess the immgen dataset."""

    translation_dict = {
        "B": "B cells",
        "DC": "Dendritic cells",
        "ILC": "Natural Killer cells",
        "Mo": "Monocytes",
        "T": "T cells",
    }

    adata.obs["celltype"] = [
        translation_dict[x.split(".")[0]]
        for x in adata.obs_names  # ["natural_language_annotation"]
    ]
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    adata.obs["batch"] = "1"

    return adata.copy()


def preprocess_tabula_sapiens(
    adata: anndata.AnnData, min_100=False
) -> anndata.AnnData:
    """Preprocess the tabula_sapiens_100_cells_per_type or the full tabula_sapiens dataset."""
    adata.obs["celltype"] = adata.obs["cell_ontology_class"]

    # replace the following labels with properly capitalized ones:

    # format: old substring, new substring, old full strings
    capitalization_list = [
        (
            "cd",
            "CD",
            [
                "cd8-positive, alpha-beta t cell",
                "cd4-positive, alpha-beta t cell",
                "cd4-positive, alpha-beta memory t cell",
                "cd8-positive, alpha-beta cytokine secreting effector t cell",
                "cd141-positive myeloid dendritic cell",
                "naive thymus-derived cd4-positive, alpha-beta t cell",
                "cd8-positive alpha-beta t cell",
                "cd4-positive alpha-beta t cell",
                "cd1c-positive myeloid dendritic cell",
                "cd4-positive helper t cell",
                "cd8-positive, alpha-beta memory t cell",
                "naive thymus-derived cd8-positive, alpha-beta t cell",
                "cd8-positive, alpha-beta cytotoxic t cell",
                "cd24 neutrophil",
                "cd8b-positive nk t cell",
            ],
        ),
        (
            "nk t cell",
            "NKT cell",
            ["type i nk t cell", "CD8b-positive nk t cell", "mature nk t cell"],
        ),
        ("nkt cell", "NKT cell", ["nkt cell"]),
        ("nk cell", "NK cell", ["nk cell"]),
        (
            "t cell",
            "T cell",
            [
                "t cell",
                "CD8-positive, alpha-beta t cell",
                "CD4-positive, alpha-beta t cell",
                "CD4-positive, alpha-beta memory t cell",
                "CD8-positive, alpha-beta cytokine secreting effector t cell",
                "naive thymus-derived CD4-positive, alpha-beta t cell",
                "CD8-positive alpha-beta t cell",
                "CD4-positive alpha-beta t cell",
                "regulatory t cell",
                "CD4-positive helper t cell",
                "CD8-positive, alpha-beta memory t cell",
                "naive thymus-derived CD8-positive, alpha-beta t cell",
                "CD8-positive, alpha-beta cytotoxic t cell",
                "naive regulatory t cell",
                "dn1 thymic pro-t cell",
            ],
        ),
        ("b cell", "B cell", ["b cell", "naive b cell", "memory b cell"]),
        ("dn4", "DN4", ["dn4 thymocyte"]),
        ("dn3", "DN3", ["dn3 thymocyte"]),
        ("dn1", "DN1", ["dn1 thymic pro-T cell"]),
        ("type ii", "type II", ["type ii pneumocyte"]),
        ("type i", "type I", ["type i NKT cell", "type i pneumocyte"]),
        ("pancreatic pp cell", "Pancreatic PP cell", ["pancreatic pp cell"]),
    ]
    for old_substring, new_substring, old_full_strings in capitalization_list:
        adata.obs["celltype"] = [
            x.replace(old_substring, new_substring) if x in old_full_strings else x
            for x in adata.obs["celltype"]
        ]

    adata.obs["celltype"] = adata.obs["celltype"].astype("category")

    adata.obs["batch"] = (
        adata.obs["donor"].astype(str) + "_" + adata.obs["method"].astype(str)
    )
    if "raw_counts" in adata.layers.keys():
        adata.X = adata.layers["raw_counts"].copy()
        # raise ValueError("Raw counts should be in adata.X, not in adata.layers['raw_counts']")

    if min_100:
        value_counts_per_celltype = adata.obs["celltype"].value_counts()
        celltypes_to_keep = value_counts_per_celltype[
            value_counts_per_celltype >= 100
        ].index.values
        adata = adata[adata.obs["celltype"].isin(celltypes_to_keep), :]

    adata.obsm["X_umap_original"] = adata.obsm["X_umap"]

    return adata.copy()


def preprocess_pancreas(adata: anndata.AnnData) -> anndata.AnnData:
    """Preprocess the pancreas dataset (manually curate cell type labels)."""
    # NOTE data does not actually contain raw counts.
    adata.obs["batch"] = adata.obs["tech"]
    adata.X = adata.layers["counts"]
    adata.var["gene_name"] = adata.var.index

    celltype_dict = {
    'gamma': 'gamma cell',
    'acinar': 'acinar cell',
    'alpha': 'alpha cell',
    'delta': 'delta cell',
    'beta': 'beta cell',
    'ductal': 'ductal cell',
    'endothelial': 'endothelial cell',
    'activated_stellate': 'activated stellate cell',
    'schwann': 'schwann cell',
    'mast': 'mast cell',
    'macrophage': 'macrophage',
    'epsilon': 'epsilon cell',
    'quiescent_stellate': 'quiescent stellate cell',
    't_cell': 'T cell'
    }

    adata.obs["celltype"] = [celltype_dict[x] for x in adata.obs["celltype"]]
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    return adata.copy()


def preprocess_human_disease(adata: anndata.AnnData) -> anndata.AnnData:
    """Preprocess the human_disease dataset."""

    adata.obs["celltype"] = adata.obs["Disease_subtype"]
    adata.obs["celltype"] = adata.obs["celltype"].astype("str").astype("category")
    adata.obs["batch"] = (
        "1"  # NOTE Could use "sra_study_acc" instead of this dummy - this will likely be strongly correlated with the celltype though..
    )
    adata.obs["Treated"] = adata.obs["Treated"].astype("bool")
    adata.obs["Treated"] = [
        "treatment status: treated" if x else "treatment status: untreated"
        for x in adata.obs["Treated"]
    ]
    adata.obs["Treated"] = adata.obs["Treated"].astype("category")
    return adata.copy()


def load_and_preprocess_dataset(
    dataset_name: str,
    read_count_table_path: str,
    obsm_paths: Optional[Mapping[str, Tuple[str, str]]] = None,
) -> anndata.AnnData:
    """Preprocess the dataset based on the provided dataset name and paths."""
    adata = anndata.read_h5ad(read_count_table_path)

    if "tabula_sapiens" in dataset_name:
        adata = preprocess_tabula_sapiens(
            adata,
            min_100="min_100" in dataset_name,
        )
    elif dataset_name == "pancreas":
        adata = preprocess_pancreas(adata)
    elif dataset_name == "human_disease":
        adata = preprocess_human_disease(adata)
    elif dataset_name == "immgen":
        adata = preprocess_immgen(adata)
    elif dataset_name == "aida":
        adata = preprocess_aida(adata)

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Very basic QC (as in zero-shot paper) NOTE: this could clash with the orig_ids check later
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=10)

    # Just to be sure and to avoid warnings later, remove the original neighbors/umaps/pcas
    if "neighbors" in adata.uns.keys():
        del adata.uns["neighbors"]
    if "X_umap" in adata.obsm.keys():
        del adata.obsm["X_umap"]
    if "X_pca" in adata.obsm.keys():
        del adata.obsm["X_pca"]

    # Load obsm if set
    if obsm_paths is not None:
        for key, (path, source_key) in obsm_paths.items():
            obj = np.load(path, allow_pickle=True)

            if dataset_name == "tabula_sapiens_well_studied_celltypes":
                # Use the orig_ids to select for the appropriate cells in obj[source_key] (adata is already filtered)

                indices = set(adata.obs.index)
                selector = [i in indices for i in obj["orig_ids"]]
                values = obj[source_key][selector]
            else:
                # Check if the original ids are still the same
                try:
                    assert (obj["orig_ids"] == adata.obs.index).all()
                except KeyError:
                    pass
                values = obj[source_key]

            adata.obsm[key] = values

    return adata.copy()
