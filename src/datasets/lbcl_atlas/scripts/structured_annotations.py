"""Extract obs metadata as structured JSON for LLM annotation generation."""
import argparse
import anndata
import pandas as pd
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

adata = anndata.read_h5ad(args.input, backed="r")

# Add study-level metadata from uns to each sample's annotation dict
study_fields = {}
for key in ["study_description", "dataset_title"]:
    if key in adata.uns:
        study_fields[key] = str(adata.uns[key])

# Rename columns for consistency with pre_training_processing prompts
obs = adata.obs.rename(columns={"disease": "disease_state"})

# Drop ID columns and technical fields
drop_fields = [c for c in obs.columns if c.endswith("_id") or c.endswith("_uuid")]
drop_fields.extend(["based_on_n_cells", "is_pseudobulk"])
drop_fields = [c for c in drop_fields if c in obs.columns]

sample_dict = obs.drop(columns=drop_fields).to_dict(orient="index")

# Clean NaN/None values and add study-level fields
for key, value in sample_dict.items():
    sample_dict[key] = {
        k: v for k, v in value.items()
        if pd.notna(v) and v not in [None, "", "<NA>", "NA", "nan"]
    }
    sample_dict[key].update(study_fields)

with open(args.output, "w") as f:
    json.dump(sample_dict, f)
