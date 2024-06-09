import anndata
import json

data = anndata.read_h5ad(snakemake.input[0], backed="r")
df = data.obs

annotations = df.apply(
    lambda row: f"{row.free_annotation} in the {row.compartment} compartment of the {row.organ_tissue}",
    axis=1,
)

with open(snakemake.output[0], "w") as f:
    json.dump(annotations.to_dict(), f, indent=4)
