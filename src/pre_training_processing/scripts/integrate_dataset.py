import anndata
import pandas as pd
import numpy as np


import json

# Load dataset
dataset = anndata.read_h5ad(snakemake.input.read_count_table, backed="r")

# Load annotations
with open(snakemake.input.processed_annotations) as f:
    annotations = json.load(f)

# Annotation keys correspond to the anndata object names (index of obs)
annotations = dataset.obs.index.map(annotations)

# Load the first annotation into the correspoding field
dataset.obs[snakemake.params.anndata_label_name] = [v[0] for v in annotations]

# Create new dataframe with index `obs.index` and as many columns as there are replicates per annotations:
annotation_replicates = pd.DataFrame(
    data=annotations.tolist(),
    index=dataset.obs.index,
    columns=[str(i) for i in range(len(annotations[0]))],
)
dataset.uns[snakemake.params.anndata_label_name + "_replicates"] = annotation_replicates

# Store the sample weights
for modality_weight_key in ["transcriptome_weights", "annotation_weights"]:
    weights = np.load(snakemake.input[modality_weight_key], allow_pickle=True)
    dataset.obs[modality_weight_key] = pd.Series(
        index=weights["orig_ids"], data=weights["weight"]
    )

# Workaround: need to reload unbacked to allow the copy
if dataset.obs[snakemake.params.anndata_label_name].isna().any():
    dataset = anndata.read_h5ad(snakemake.input.read_count_table)
    dataset.obs[snakemake.params.anndata_label_name] = dataset.obs.index.map(
        [v[0] for v in annotations]
    )
    dataset.uns[
        snakemake.params.anndata_label_name + "_replicates"
    ] = annotation_replicates

    # Filter dataset to only contain annotated cells. need to copy to be able to write (bug)
    dataset = dataset[dataset.obs[snakemake.params.anndata_label_name].notnull()].copy()

# Save dataset
dataset.write(snakemake.output[0])
