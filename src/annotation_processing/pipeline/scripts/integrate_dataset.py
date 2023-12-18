import anndata
import json

# Load dataset
dataset = anndata.read_h5ad(snakemake.input.read_count_table, backed="r")

# Load annotations
with open(snakemake.input.processed_annotations) as f:
    annotations = json.load(f)

# Annotation keys correspond to the anndata object names (index of obs)
dataset.obs[snakemake.params.anndata_label_name] = dataset.obs.index.map(annotations)

if dataset.obs[snakemake.params.anndata_label_name].isna().any():
    # Workaround: need to reload unbacked to allow the copy
    dataset = anndata.read_h5ad(snakemake.input.read_count_table)
    dataset.obs[snakemake.params.anndata_label_name] = dataset.obs.index.map(
        annotations
    )

    # filter dataset to only contain annotated cells. need to copy to be able to write (bug)
    dataset = dataset[dataset.obs[snakemake.params.anndata_label_name].notnull()].copy()

# Save dataset
dataset.write(snakemake.output[0])
