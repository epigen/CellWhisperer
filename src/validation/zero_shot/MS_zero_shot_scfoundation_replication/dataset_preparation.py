import scanpy as sc
import anndata
from cellwhisperer.config import get_path
import numpy as np
import glob
import pandas as pd
from utils import TABSAP_WELLSTUDIED_COLORMAPPING


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


def preprocess_covid(adata: anndata.AnnData) -> anndata.AnnData:
    """Preprocess the covid_train dataset."""
    # Data is raw counts
    adata.obs["batch"] = adata.obs["batch_id"].astype("category")
    adata.obs["smoking"] = "smoking status: " + adata.obs["smoking"].astype(str)
    adata.obs["smoking"] = adata.obs["smoking"].replace("smoking status: nan", np.nan)
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    return adata.copy()

def load_and_process_liao_covid() -> anndata.AnnData:
    """Preprocess the liao_covid dataset."""

    # TODO
    file_path = '/msc/home/q56ppene/cellwhisperer/cellwhisperer/resources/liao_covid/*_filtered_feature_bc_matrix.h5'

    adatas={}
    for file in glob.glob(file_path):
        samplename=file.split('/')[-1].split('_')[1]
        adatas[samplename]=sc.read_10x_h5(file)
        adatas[samplename].var_names_make_unique()
        adatas[samplename].obs['sample']=samplename
        adatas[samplename].obs['batch']=samplename
        adatas[samplename].obs.index=[x.replace("1",samplename) for x in adatas[samplename].obs.index]
    adata=anndata.concat(adatas.values())

    # read the all.cell.annotation.meta.txt
    df=pd.read_csv('/msc/home/q56ppene/cellwhisperer/cellwhisperer/resources/liao_covid/all.cell.annotation.meta.txt',sep='\t')
    df=df.set_index('ID')
    df.index=[f'{x.split("_")[0]}-{samplename}' for x, samplename in zip(df.index,df['sample'])]
    df=df.loc[[x for x in df.index if x in adata.obs.index]]
    adata=adata[df.index]
    adata.obs['celltype']=df['celltype']
    adata.obs['sample_new']=df['sample_new']

    # TODO improve classification of the cytokine levels?
    # Plot 3: Il-6, Il-8 and IL-1beta
    # 0: Nothing in plot  3
    # 1: Low in plot 3
    # 2: high (around 5000 or more) in plot 3
    cytokine_levels_dict={
        "M1": 0,
        "M2": 0,
        "M3": 0,
        "S1": 1,
        "S7": 1,
        "S6-1": 2,
        "S6-2": 1,
        "S3": 1,
        "S4": 2,
        "S5": 1,
        "S10": 2,
        "S2-1":2,
        "S2-2":2,
        "S2-3":2,
        "S8-1":1,
        "S8-2":2,
        "S8-3":2,
        "S9-1":2,
        "S9-2":2,
    }
    adata.obs['cytokine_level']=[cytokine_levels_dict[x] for x in adata.obs['sample_new']]

    return adata



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


def load_and_preprocess_dataset(dataset_name: str) -> anndata.AnnData:
    """Preprocess the dataset based on the provided dataset name."""

    if dataset_name == "liao_covid":
        adata = load_and_process_liao_covid()
    else:
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
