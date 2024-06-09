import anndata
import pandas as pd
import json

adata = anndata.read_h5ad(snakemake.input[0], backed="r")

adata.obs.rename(
    columns={
        "abstract": "study_description",
        "dataset_title": "study_title",
        "disease": "disease_state",
        "sample": "sample_source_name",  # don't confuse. for us 'sample' is the single cell
    },
    inplace=True,
)

drop_fields = [
    col for col in adata.obs.columns if col.endswith("_id") or col.endswith("_uuid")
]
drop_fields.extend(
    ["based_on_n_cells", "assay", "batch", "predicted_doublets", "is_primary_data"]
)

sample_dict = adata.obs.drop(columns=drop_fields).to_dict(orient="index")
# Delete all fields with None/NaN values
for key, value in sample_dict.items():
    sample_dict[key] = {
        k: v
        for k, v in value.items()
        if pd.notna(v) and v not in [None, "", "<NA>", "NA", "nan"]
    }

# Write the JSON string to a file
with open(snakemake.output[0], "w") as f:
    json.dump(sample_dict, f)
