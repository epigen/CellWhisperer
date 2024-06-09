import anndata
import pandas as pd
import numpy as np
import json
from scipy import sparse
import scanpy as sc
import os
import logging
from utils import get_abstract
import sys
import gc


def is_count_matrix(matrix, threshold=1e-6, max_entries=10000):
    # Limit the number of entries to check
    data_to_check = matrix.data[:max_entries]

    # Check if the fractional part is close to zero for the selected entries
    fractional_parts = np.modf(data_to_check)[0]

    if np.all(fractional_parts < threshold):
        return True
    else:
        return False


# Prepare anndata
adata = anndata.read_h5ad(str(snakemake.input.adata))

if adata.raw is not None:
    adata.X = adata.raw.X
    adata.raw = None
else:
    if not np.issubdtype(adata.X.dtype, np.integer):
        assert is_count_matrix(
            adata.X
        ), f"adata object dtype must be integer or pass is_count_matrix test, but is {adata.X.dtype}. X: {adata.X[:50,:50]}"

assert adata.obs.is_primary_data.dtype == "bool"
adata = adata[adata.obs.is_primary_data]  # .copy()
sc.pp.filter_cells(adata, min_genes=snakemake.params.min_genes_per_cell)
gc.collect()  # call this everywhere to prevent apparent memory leaks
adata.obs["is_primary_data"] = "True"  # neeeds to be a string for write_h5ad

# Filter out columns with too many distinct values
n_unique_vals_per_obs_col = adata.obs.nunique()
relevant_obs_cols = n_unique_vals_per_obs_col[
    (n_unique_vals_per_obs_col < snakemake.params.max_unique_values_per_obs_columns)
    & (n_unique_vals_per_obs_col < adata.obs.shape[0])
].index.to_list()
relevant_obs_cols = [
    col for col in relevant_obs_cols if col not in snakemake.params.ignore_cols
]
relevant_obs_cols = sorted(
    list(set(relevant_obs_cols + snakemake.params.always_use_cols))
)
unique_obs_combinations_df = (
    adata.obs[relevant_obs_cols].drop_duplicates().reset_index(drop=True)
)


# Prepare dataset metadata
census_dataset = pd.read_csv(str(snakemake.input.census_datasets_csv), index_col=0).loc[
    snakemake.wildcards.dataset_id
]

adatas_processed = []
for i, row in unique_obs_combinations_df.iterrows():
    # need to take care of nan values, because np.nan!=np.nan
    conditions = [
        (adata.obs[col] == row[col])
        if row[col] == row[col]
        else adata.obs[col] != adata.obs[col]
        for col in relevant_obs_cols
    ]
    mask = pd.concat(conditions, axis=1).all(axis=1)
    adata_this_condition = adata[mask]  # .copy()

    if adata_this_condition.obs.shape[0] < 1:
        logging.error(
            f"No cells found in anndata: {adata_this_condition}, row: {row}, counts: {n_unique_vals_per_obs_col}"
        )
        raise ValueError

    # Gene length normalization - skipped for now at least
    # if "Smart-seq" in adata.obs["assay"].values[0]:
    #     gene_lengths = adata_this_condition.var[["feature_length"]].to_numpy()
    #     # https://chanzuckerberg.github.io/cellxgene-census/notebooks/analysis_demo/comp_bio_normalizing_full_gene_sequencing.html
    #     # https://chanzuckerberg.github.io/cellxgene-census/notebooks/analysis_demo/comp_bio_data_integration_scvi.html#Gene-length-normalization-of-Smart-Seq2-data.
    #     adata_this_condition.X = csc_matrix (((adata_this_condition.X.T / gene_lengths).T).ceil())
    # else:
    #     assert adata.obs[adata.obs["assay"].str.lower().str.contains("smart")].shape[0]==0

    if not i % 10:
        print(
            f"Processing sub-batch {i}  of {len(unique_obs_combinations_df)} ({adata_this_condition.obs.shape[0]} cells)"
        )
    # list all objects and their memory usage:
    # print([(name, sys.getsizeof(value)) for name, value in locals().items()])
    # import psutil
    # print(f"Memory usage: {psutil.virtual_memory().percent}%")

    # Pseudobulk

    pseudobulk_X = sparse.csc_matrix(
        np.mean(adata_this_condition.X, axis=0).reshape(1, -1)
    )
    pseudobulk_adata = anndata.AnnData(
        X=pseudobulk_X,
        obs=pd.DataFrame(adata_this_condition.obs.copy().iloc[0]).T[relevant_obs_cols],
        var=adata_this_condition.var.copy(),
    )
    pseudobulk_adata.obs.index = [f"census_{snakemake.wildcards.dataset_id}_pseudobulk"]
    pseudobulk_adata.obs["is_pseudobulk"] = "True"
    pseudobulk_adata.obs["replicate"] = "pseudobulk"
    pseudobulk_adata.obs["based_on_n_cells"] = adata_this_condition.shape[0]
    adatas_processed.append(pseudobulk_adata)

    # Random subsets of cells
    np.random.seed(42)
    cell_indices = adata_this_condition.obs.index
    subsampled_indices = np.random.choice(
        cell_indices,
        size=min(snakemake.params.n_single_cells, len(cell_indices)),
        replace=False,
    )
    subsampled_adata = adata_this_condition[subsampled_indices]  # .copy()
    subsampled_adata.obs.index = [
        f"census_{snakemake.wildcards.dataset_id}_{i}_randomCell{x}"
        for x in range(subsampled_adata.obs.shape[0])
    ]
    subsampled_adata.obs["is_pseudobulk"] = "False"
    subsampled_adata.obs["replicate"] = [
        str(x) for x in range(subsampled_adata.obs.shape[0])
    ]
    subsampled_adata.obs["based_on_n_cells"] = 1
    adatas_processed.append(subsampled_adata)

    del adata_this_condition
    del pseudobulk_adata
    del subsampled_adata
    del pseudobulk_X
    gc.collect()

adata_n_cells = adata.obs.shape[0]
del adata
gc.collect()
adata_processed = anndata.concat(adatas_processed)
dataset_info_dict = {}

adata_processed.uns["dataset_title"] = census_dataset["dataset_title"]
adata_processed.uns["dataset_id"] = snakemake.wildcards.dataset_id
adata_processed.uns["collection_name"] = census_dataset["collection_name"]
adata_processed.uns["collection_doi"] = census_dataset["collection_doi"]
adata_processed.uns["abstract"] = get_abstract(census_dataset["collection_doi"])
adata_processed.uns["census_version"] = snakemake.params.census_version
adata_processed.uns["n_primary_cells_in_complete_dataset"] = int(adata_n_cells)
adata_processed.uns[
    "obs_columns_used_for_splitting"
] = relevant_obs_cols  # and this to find out why groups are too small
adata_processed.obs = (
    adata_processed.obs.infer_objects()
)  # Otherwise sometimes we get an error "Can't implicitly convert non-string objects to strings"
adata_processed.write_h5ad(snakemake.output[0])

dataset_info_dict = adata_processed.uns.copy()
dataset_info_dict["adata_file_name"] = os.path.basename(snakemake.output[0])
dataset_info_dict["n_transcriptomes"] = adata_processed.obs.shape[0]
dataset_info_dict["n_pseudobulks"] = int(
    adata_processed.obs["is_pseudobulk"].value_counts()["True"]
)
dataset_info_dict["n_random_cell_samples"] = int(
    adata_processed.obs["is_pseudobulk"].value_counts()["False"]
)
dataset_info_dict["n_cells_per_group"] = adata_processed.obs[
    adata_processed.obs["is_pseudobulk"] == "True"
][
    "based_on_n_cells"
].values.tolist()  # this may be useful to spot datasets were the groups are too small
dataset_info_dict["obs_columns_all"] = adata_processed.obs.columns.to_list()

with open(snakemake.output[1], "w") as f:
    json.dump(dataset_info_dict, f)
