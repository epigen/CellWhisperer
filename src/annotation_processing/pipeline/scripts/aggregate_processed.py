import json
from collections import defaultdict
from pathlib import Path

annotations = {}
for annotation_fn in snakemake.input.processed_annotations:
    annotation_id = Path(annotation_fn).stem
    annotations[annotation_id] = Path(annotation_fn).read_text()

with open(snakemake.output[0], "w") as f:
    json.dump(annotations, f, indent=4)
