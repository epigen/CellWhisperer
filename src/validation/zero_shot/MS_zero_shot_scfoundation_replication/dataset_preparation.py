import scanpy as sc
import anndata
from cellwhisperer.config import get_path, config
from typing import Tuple
import numpy as np


def load_dataset(dataset_name: str) -> anndata.AnnData:
    """Load the dataset's adata based on the provided dataset name."""
    return anndata.read_h5ad(
        get_path(
            [
                "paths",
                "read_count_table"
                if not dataset_name in ["daniel", "immgen"]
                else "full_dataset",
            ],
            dataset=dataset_name,
        )
    )


def preprocess_immgen(adata: anndata.AnnData) -> anndata.AnnData:
    """Preprocess the immgen dataset."""
    translation_dict = {
        "Activated regulatory T cells from human blood": "T Cells",
        "Effector memory CD4 T cells that express CD3, CD4, but not CD45RA or CD62L from human blood": "T Cells",
        "Effector memory CD8 T cells that express CD3, CD8, but not CD45RA or CD62L from human blood": "T Cells",
        "Immature natural killer cells from the innate lymphoid cell group with high expression of CD56 but not CD16 from human blood": "Natural Killer Cells",
        "MAIT cells that express CD4 from human blood": "T Cells",
        "MAIT cells that express CD8 from human blood": "T Cells",
        "Mature natural killer cells from the innate lymphoid cell group with low expression of CD56, high expression of CD16, and no expression of CD57 from human blood": "Natural Killer Cells",
        "Memory B cells that do not express IgD but do express CD27 and not CD38 from human blood": "B Cells",
        "Memory natural killer cells from the innate lymphoid cell group with low expression of CD56, high expression of CD16, and high expression of CD57 from human blood": "Natural Killer Cells",
        "Monocytes that express CD14 from human blood": "Monocytes",
        "Monocytes that express CD16 from human blood": "Monocytes",
        "Naive B cells that express IgD but not CD27 from human blood": "B Cells",
        "Naive CD4 T cells that express CD3, CD4, CD45RA, and CD62L from human blood": "T Cells",
        "Naive CD8 T cells that express CD3, CD8, CD45RA, and CD62L from human blood": "T Cells",
        "NKT cells that express Va24 from human blood": "Natural Killer T Cells",
        "Resting regulatory T cells from human blood": "T Cells",
        "Transitional B cells that express both IgD and CD27 from human blood": "B Cells",
        "Type 1 dendritic cells that express CD141 from human blood": "Dendritic Cells",
        "Type 5 dendritic cells that express AXL and SIGLEC6 from human blood": "Dendritic Cells",
        "Type 6 dendritic cells that express CD123 from human blood": "Dendritic Cells",
    }

    adata.obs["celltype"] = [
        translation_dict[x] for x in adata.obs["natural_language_annotation"]
    ]
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    adata.obs["batch"] = "1"

    return adata.copy()


def preprocess_tabula_sapiens(
    adata: anndata.AnnData, well_studied_only=False, min_100=False
) -> anndata.AnnData:
    """Preprocess the tabula_sapiens_100_cells_per_type or the full tabula_sapiens dataset."""
    adata.obs["celltype"] = adata.obs["cell_ontology_class"]
    adata.obs["batch"] = (
        adata.obs["donor"].astype(str) + "_" + adata.obs["method"].astype(str)
    )
    adata.X = adata.layers[
        "raw_counts"
    ]  # TODO remove later once we're sure we store raw counts in X by default already
    if well_studied_only:
        adata = adata[
            adata.obs["celltype"].isin(config["top20_lung_liver_blood_celltypes"]), :
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
    # NOTE data does not seem to be raw counts - even the "counts" layer.
    # I checked but getting the raw counts would be quite tricky, since the
    # dataset consists of results from multiple techniques, some of which require normalization etc.
    adata.obs["batch"] = adata.obs["tech"]
    adata.X = adata.layers["counts"]
    adata.var["gene_name"] = adata.var.index
    adata.obs["celltype"] = [x.replace("_", " ") for x in adata.obs["celltype"]]
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    return adata.copy()


def preprocess_covid(adata: anndata.AnnData) -> anndata.AnnData:
    """Preprocess the covid_train dataset."""
    # Data is raw counts
    adata.obs["batch"] = adata.obs["batch_id"].astype("category")
    adata.obs["smoking"] = "smoking status: " + adata.obs["smoking"].astype(str)
    adata.obs["smoking"] = adata.obs["smoking"].replace("smoking status: nan", np.nan)
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    return adata.copy()


def preprocess_immune_330k(adata: anndata.AnnData) -> anndata.AnnData:
    """Preprocess the immune_330k dataset."""
    # TODO data does not seem to be raw counts
    adata.obs["batch"] = (
        adata.obs["Organ"].astype(str) + "_" + adata.obs["Chemistry"].astype(str)
    )
    adata.obs["celltype"] = adata.obs["Manually_curated_celltype"]
    adata.var["gene_name"] = adata.var.index

    organ_dict = {
        "BLD": "Blood",
        "BMA": "Bone Marrow",
        "CAE": "Caecum",
        "DUO": "Duodenum",
        "ILE": "Ileum",
        "JEJEPI": "Jejunum",
        "JEJLP": "Jejunum",
        "LIV": "Liver",
        "LLN": "Lung-draining Lymph Node",
        "LNG": "Lung",
        "MLN": "Mesenteric Lymph Node",
        "OME": "Omentum",
        "SCL": "Signmoid Colon",
        "SKM": "Skeletal Muscle",
        "SPL": "Spleen",
        "TCL": "Transverse Colon",
        "THY": "Thymus",
    }
    adata.obs["organ_full_name"] = adata.obs["Organ"].map(organ_dict)
    return adata.copy()


def preprocess_daniel(adata: anndata.AnnData) -> anndata.AnnData:
    """Preprocess the daniel dataset."""
    daniel_dedup_colname = "cluster_assignment_daniel_normally_deduplicated_dmis-lab_biobert-v1.1_CLS_pooling"
    adata.obs["celltype"] = adata.obs[daniel_dedup_colname]
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


def load_and_preprocess_dataset(dataset_name: str) -> anndata.AnnData:
    """Preprocess the dataset based on the provided dataset name."""

    adata = load_dataset(
        dataset_name.replace("_well_studied_celltypes", "").replace("_min_100", "")
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
    elif "immune_330k" in dataset_name:
        adata = preprocess_immune_330k(adata)
    elif dataset_name == "daniel":
        adata = preprocess_daniel(adata)
    elif "covid" in dataset_name:
        adata = preprocess_covid(adata)
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
