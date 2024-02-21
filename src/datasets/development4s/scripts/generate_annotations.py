import anndata
import json

adata = anndata.read_h5ad(snakemake.input.read_count_table, backed="r")
annotations = adata.obs.to_json(snakemake.output.structured, orient="index")

adata.obs["dev_stage_string"] = adata.obs.developmental_stage.apply(
    {1: "pre-implantation", 2: "(pre-)gastrulation", 3: "post-gastrulation"}.get
)
annotations = adata.obs.apply(
    lambda row: f"A {row.anno_og} cell, also identified as {row.anno_new.replace('_', ', ')} from {row.anno_time} ({row.developmental_stage} stage)",
    axis=1,
)

with open(snakemake.output.processed, "w") as f:
    json.dump(annotations.to_dict(), f, indent=4)
