import json
from collections import defaultdict
from pathlib import Path
import pandas as pd
import re
import logging

df = pd.concat([pd.read_csv(path) for path in snakemake.input.yaml_splits])

assert df.annotation.isna().sum() == 0, "Found NaNs in the annotations"
assert df.annotation.str.contains("\n").sum() == 0, "Found newlines in the annotations"
assert (
    df.annotation.str.len().min() > 10
), "Found annotations with less than 10 characters"

# Create a JSON dict, with keys corresponding to sample_id and values corresponding to annotation (for replicate 0 only)
annotations_dict = (
    df[df.replicate.astype(int) == 0].set_index("sample_id")["annotation"].to_dict()
)

with open(snakemake.output["single"], "w") as f:
    json.dump(annotations_dict, f)

# Create a JSON dict, with keys corresponding to sample_id and values corresponding to a list of annotations (sorted by replicate)

# Sort the dataframe by 'sample_id' and 'replicate'
df_sorted = df.sort_values(by=["sample_id", "replicate"])

# Group by 'sample_id' and aggregate the 'annotation' into a list
grouped_annotations = df_sorted.groupby("sample_id")["annotation"].apply(list).to_dict()

# Save the grouped annotations to a JSON file
with open(snakemake.output["multi"], "w") as f:
    json.dump(grouped_annotations, f)
