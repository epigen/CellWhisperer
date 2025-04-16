import shutil
from pathlib import Path

from huggingface_hub import login, snapshot_download
import pandas as pd
import scanpy as sc
import datasets
import numpy as np
import math

SCALE_FACTOR = 500
VIEW_X = 500  # approx. viewport width


def augment_metadata(adata):
    adata.layers["counts"] = (np.exp(adata.X) * 10).astype(
        int
    )  # approximate raw read counts (as required by scFMs)
    adata.var["gene_name"] = adata.var.index

    # Format for spatial whisperer.
    image_data = adata.uns["20x_slide"]
    extent_x = image_data.shape[0] * 2
    extent_y = image_data.shape[1] * 2
    adata.uns["image_extents"] = np.array([extent_x, extent_y])

    # this dataset contains very large images, so we will already shrink them to reduce loading times.
    # configureable by SCALE_FACTOR above, but should maybe go into snakemake.
    original_aspect = image_data.shape[0] / image_data.shape[1]
    sample_x = max(math.floor(image_data.shape[0] / (SCALE_FACTOR * VIEW_X)), 1)
    sample_y = max(
        math.floor(image_data.shape[1] / (SCALE_FACTOR * (VIEW_X * original_aspect))), 1
    )

    down_sampled = image_data[::sample_x, ::sample_y]
    adata.uns["20x_slide"] = down_sampled

    # Copy the pixel coordinates into the spatial layout and make this the default.
    adata.obsm["X_spatial"] = pd.concat(
        (adata.obs["x_pixel"], adata.obs["y_pixel"]), axis=1
    ).to_numpy()

    adata.uns["default_embedding"] = "X_spatial"

    return adata  # it's anyways inplace


login(token=snakemake.params.huggingface_token)

# download the dataset
snapshot_download(
    "nonchev/TCGA_digital_spatial_transcriptomics",
    local_dir=snakemake.output.download_dir,
    cache_dir=snakemake.output.download_dir,
    local_dir_use_symlinks=False,
    allow_patterns=[snakemake.params.sample_id_small],
    repo_type="dataset",
)

adata = sc.read_h5ad(
    Path(snakemake.output.download_dir) / snakemake.params.sample_id_small
)
augment_metadata(adata)

adata.write_h5ad(snakemake.output.dataset_small)
