import shutil
from pathlib import Path

from huggingface_hub import login, snapshot_download
import pandas as pd
import scanpy as sc
import datasets
import numpy as np
import math
from PIL import Image
import random
import logging

logging.basicConfig(level=logging.INFO)

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


def crop_tile(image, x, y, size):
    """Crops a tile from a numpy image."""
    img_height, img_width = image.shape[:2]
    x = max(0, min(x, img_width - size))
    y = max(0, min(y, img_height - size))
    tile = image[y : y + size, x : x + size]
    return tile


def generate_example_patches(adata, num_patches=3):
    """Generate example patches from the dataset for QC reporting."""
    image = adata.uns["20x_slide"]
    patch_size = 224  # Standard patch size

    # Sample random spots for patch generation
    random.seed(42)
    sample_indices = random.sample(
        range(len(adata.obs)), min(num_patches, len(adata.obs))
    )

    patches = []
    for idx in sample_indices:
        x = int(adata.obs.iloc[idx]["x_pixel"])
        y = int(adata.obs.iloc[idx]["y_pixel"])
        patch = crop_tile(image, x, y, patch_size)
        patches.append(patch)

    return patches


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

# Generate example patches for report
logging.info(
    f"Generating {len(snakemake.output.report_patches)} example patches for QC"
)
patches = generate_example_patches(adata, len(snakemake.output.report_patches))

for i, patch in enumerate(patches):
    patch_image = Image.fromarray(patch.astype(np.uint8))
    output_path = snakemake.output.report_patches[i]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    patch_image.save(output_path)
    logging.info(f"Saved example patch to {output_path}")

adata.write_h5ad(snakemake.output.dataset_small)
