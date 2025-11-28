"""
Create quality control patches from existing h5ad files.

This script generates example patches from the created h5ad files for quality control reporting.
It reads a few h5ad files and extracts patches for visualization.
"""

import json
import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path
import openslide
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

print("Creating quality control patches from h5ad files...")

# Get list of created h5ad files from input
h5ad_files = [Path(f) for f in snakemake.input.h5ads]
logging.info(f"Found {len(h5ad_files)} h5ad files")

example_patches = []
max_patches_needed = len(snakemake.output.report_patches)

for h5ad_file in h5ad_files[:max_patches_needed]:  # Only process as many as we need
    # Load h5ad file
    adata = ad.read_h5ad(h5ad_file)

    if adata.n_obs == 0:
        logging.info(f"Skipping empty h5ad file: {h5ad_file}")
        continue

    # Get WSI path and load WSI
    wsi_path = Path(adata.uns["image_path"])
    if not wsi_path.exists():
        logging.warning(f"WSI file not found: {wsi_path}")
        continue

    wsi = openslide.OpenSlide(str(wsi_path))

    # Get first patch coordinates
    first_obs = adata.obs.iloc[0]
    x_center, y_center = first_obs["x_pixel"], first_obs["y_pixel"]

    # Convert center coordinates to top-left for OpenSlide
    half_patch = snakemake.params.patch_size // 2
    x_topleft = x_center - half_patch
    y_topleft = y_center - half_patch

    logging.info(
        f"Extracting QC patch from {h5ad_file.name} at center coords ({x_center},{y_center})"
    )

    # Extract patch
    patch = wsi.read_region(
        (x_topleft, y_topleft),
        0,
        (snakemake.params.patch_size, snakemake.params.patch_size),
    )

    # Convert RGBA to RGB
    patch_rgb = Image.new("RGB", patch.size, (255, 255, 255))
    patch_rgb.paste(
        patch,
        mask=patch.split()[3] if len(patch.split()) == 4 else None,
    )

    # Save patch
    output_patch_path = snakemake.output.report_patches[len(example_patches)]
    Path(output_patch_path).parent.mkdir(parents=True, exist_ok=True)
    patch_rgb.save(output_patch_path)
    example_patches.append(output_patch_path)
    logging.info(f"Saved QC patch to {output_patch_path}")

    wsi.close()

    if len(example_patches) >= max_patches_needed:
        break

# # Create any remaining placeholder patches if we don't have enough
# while len(example_patches) < max_patches_needed:
#     placeholder_path = snakemake.output.report_patches[len(example_patches)]
#     Path(placeholder_path).parent.mkdir(parents=True, exist_ok=True)

#     # Create a simple placeholder image
#     placeholder_img = Image.new(
#         "RGB",
#         (snakemake.params.patch_size, snakemake.params.patch_size),
#         (200, 200, 200),
#     )
#     placeholder_img.save(placeholder_path)
#     example_patches.append(placeholder_path)
#     logging.info(f"Created placeholder patch at {placeholder_path}")

logging.info(f"QC patch generation completed. Created {len(example_patches)} patches")
