import anndata
import pandas as pd
import numpy as np
import logging


import json

# Load dataset
dataset = anndata.read_h5ad(snakemake.input.read_count_table, backed="r")  # type: ignore [reportUndefinedVariable]

# # Load GSVA data
# def load_gsva_data(top_n=50):
#     """
#     Identify the top_n pathways for each sample (by GSVA score) and store them as top50 categorical dataframe.
#     I also tried a sparse type dataframe, but it unfortunately did not store with the anndata object's h5py .
#     """
#     gsva = pd.read_csv(snakemake.input.gsva_csv, index_col=0)
#     gsva.index.name = "pathway"

#     # NOTE: this operation might be slow/big
#     top_n_pathways = gsva.apply(lambda x: x.nlargest(top_n).index, axis=0)

#     # Convert to categorical
#     top_n_pathways = top_n_pathways.apply(
#         lambda x: pd.Categorical(x, categories=gsva.index)
#     )

#     # Select the scores for each of the top_n_pathways from `gsva`
#     scores = top_n_pathways.apply(lambda sample: gsva.loc[sample, sample.name])

#     top_n_pathways.index = top_n_pathways.index.map(
#         str
#     )  # required to be storable as anndata

#     return top_n_pathways.T, scores.T


# # Add GSVA data to the dataset
# dataset.obsm["top_50_gsva_names"], dataset.obsm["top_50_gsva_scores"] = load_gsva_data()

# Load annotations
with open(snakemake.input.processed_annotations) as f:  # type: ignore [reportUndefinedVariable]
    annotations = json.load(f)

# Annotation keys correspond to the anndata object names (index of obs)
annotations = dataset.obs.index.map(annotations)

# Load the first annotation into the correspoding field
dataset.obs[snakemake.params.anndata_label_name] = [v[0] for v in annotations]  # type: ignore [reportUndefinedVariable]

# Create new dataframe with index `obs.index` and as many columns as there are replicates per annotations:
annotation_replicates = pd.DataFrame(
    data=[v[1:] for v in annotations],
    index=dataset.obs.index,
    columns=[str(i) for i in range(1, len(annotations[0]))],
)
dataset.obsm[
    snakemake.params.anndata_label_name + "_replicates"  # type: ignore [reportUndefinedVariable]
] = annotation_replicates

# Store the sample weights
for modality_weight_key in ["transcriptome_weights", "annotation_weights"]:
    weights = np.load(snakemake.input[modality_weight_key], allow_pickle=True)  # type: ignore [reportUndefinedVariable]
    dataset.obs[modality_weight_key] = pd.Series(
        index=weights["orig_ids"], data=weights["weight"]
    )

# Workaround: need to reload unbacked to allow the copy
if dataset.obs[snakemake.params.anndata_label_name].isna().any():  # type: ignore [reportUndefinedVariable]
    logging.warning(
        "Some cells are not annotated. Removing them from the dataset. This may lead to inconsistencies with the rest of the pipeline"
    )
    dataset = anndata.read_h5ad(snakemake.input.read_count_table)  # type: ignore [reportUndefinedVariable]
    dataset.obs[snakemake.params.anndata_label_name] = dataset.obs.index.map(  # type: ignore [reportUndefinedVariable]
        [v[0] for v in annotations]
    )
    dataset.obsm[
        snakemake.params.anndata_label_name + "_replicates"  # type: ignore [reportUndefinedVariable]
    ] = annotation_replicates

    # Filter dataset to only contain annotated cells. need to copy to be able to write (bug)
    dataset = dataset[dataset.obs[snakemake.params.anndata_label_name].notnull()].copy()  # type: ignore [reportUndefinedVariable]

# Save dataset
dataset.write(snakemake.output[0])  # type: ignore [reportUndefinedVariable]
