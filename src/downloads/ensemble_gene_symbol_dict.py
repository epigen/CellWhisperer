import scanpy as sc

from os import popen

import yaml
from pathlib import Path


PROJECT_DIR = Path(popen("git rev-parse --show-toplevel").read().strip())
with open(PROJECT_DIR / "config.yaml") as f:
    config = yaml.safe_load(f)

# Assuming gene symbol names. Use biomart to get ensembl_ids
# use_cache=False to avoid the error sqlite3.OperationalError: database is locked
annot = sc.queries.biomart_annotations(
    "hsapiens", ["ensembl_gene_id", "external_gene_name"], use_cache=False
).set_index("external_gene_name")

annot_drop_dups = annot.reset_index().drop_duplicates(subset="external_gene_name")
annot_drop_dups = annot_drop_dups.set_index("external_gene_name")

annot_drop_dups.to_csv(PROJECT_DIR / config["paths"]["ensembl_gene_symbol_map"])
