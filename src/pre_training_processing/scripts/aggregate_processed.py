import json
from collections import defaultdict
from pathlib import Path
import pandas as pd
import re


annotations = {}

all_annots = snakemake.input.processed_annotations


def extract_fields_from_path(path):
    """Extract fields replicate and sample_id from a path

    PROCESSED_FILE = PROJECT_DIR / "results" / "pre_training_processing" / "processed" / "dataset" / "{replicate}" / "{sample_id}.txt"
    """
    match = re.match(r".*/(?P<replicate>\d+)/(?P<sample_id>.+).txt", path)
    return match.groupdict()


df = pd.DataFrame(
    [extract_fields_from_path(path) for path in all_annots], index=all_annots
)
df["replicate"] = df["replicate"].astype(int)
df["annotation"] = df.index.map(lambda s: Path(s).read_text())

# Create a JSON dict, with keys corresponding to sample_id and values corresponding to annotation (for replicate 0 only)

annotations_dict = df[df.replicate == 0].set_index("sample_id")["annotation"].to_dict()

with open(snakemake.output["single"], "w") as f:
    json.dump(annotations_dict, f)


### Create a JSON dict, with keys corresponding to sample_id and values corresponding to a list of annotations (sorted by replicate)

# Sort the dataframe by 'sample_id' and 'replicate'
df_sorted = df.sort_values(by=["sample_id", "replicate"])

# Group by 'sample_id' and aggregate the 'annotation' into a list
grouped_annotations = df_sorted.groupby("sample_id")["annotation"].apply(list).to_dict()

# Save the grouped annotations to a JSON file
with open(snakemake.output["multi"], "w") as f:
    json.dump(grouped_annotations, f)
