import shutil
from pathlib import Path

from huggingface_hub import login, snapshot_download
import pandas as pd
import scanpy as sc
import datasets
import numpy as np

login(token=snakemake.params.huggingface_token)
snapshot_download("nonchev/TCGA_digital_spatial_transcriptomics",
                  local_dir=snakemake.output.download_dir,
                  allow_patterns=[snakemake.params.sample_id_small],
                  repo_type="dataset")

adata = sc.read_h5ad(Path(snakemake.output.download_dir) / snakemake.params.sample_id_small)
adata.layers["counts"] = (np.exp(adata.X) * 10).astype(int)  # approximate raw read counts (as required by scFMs)
adata.var["gene_name"]= adata.var.index
adata.write_h5ad(snakemake.output.dataset_small)
