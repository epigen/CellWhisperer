import anndata
import json

# Load dataset
dataset = anndata.read_h5ad(snakemake.input.read_count_table)

# Load annotations
with open(snakemake.input.processed_annotations) as f:
    annotations = json.load(f)

# Annotation keys correspond to the anndata object names (index of obs)
dataset.obs[snakemake.params.anndata_label_name] = dataset.obs.index.map(annotations)

# filter dataset to only contain annotated cells
dataset = dataset[dataset.obs[snakemake.params.anndata_label_name].notnull()]

# Save dataset
dataset.write(snakemake.output[0])
