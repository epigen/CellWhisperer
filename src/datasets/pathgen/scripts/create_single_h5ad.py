"""
Create a single h5ad file for one WSI from PathGen data.

This script processes one WSI at a time, extracting patches and metadata
to create an AnnData object following the quilt1m structure.
Uses individual WSI metadata files for maximum efficiency.
"""

import json
import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path
import openslide
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

file_id = snakemake.wildcards.file_id
print(f"Creating h5ad file for WSI: {file_id}")

# Load individual WSI metadata (much more efficient than loading full filtered metadata)
with open(snakemake.input.wsi_metadata, "r") as f:
    entries = json.load(f)

if not entries:
    logging.warning(f"No metadata entries found for WSI {file_id}")
    # Create empty h5ad file
    obs_df = pd.DataFrame(columns=["patch_id", "x_pixel", "y_pixel", "natural_language_annotation", "wsi_id", "file_id"])
    var_data = pd.DataFrame(index=[])
    X = np.empty((0, 0))
    adata = ad.AnnData(X=X, obs=obs_df, var=var_data)
    
    # Ensure output directory exists
    Path(snakemake.output.h5ad).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(snakemake.output.h5ad)
    exit(0)

logging.info(f"Loaded {len(entries)} entries for WSI {file_id} from individual metadata file")

# Load WSI
wsi_path = Path(snakemake.input.wsi_file)
logging.info(f"Processing WSI: {wsi_path}")

wsi = openslide.OpenSlide(str(wsi_path))
wsi_width, wsi_height = wsi.dimensions

# Create observation dataframe for this WSI
obs_data = []
valid_entries = 0

for entry in entries:
    position = entry["position"]
    # PathGen coordinates are patch centers; original JSON contains top-left from TCGA extraction
    x_topleft, y_topleft = int(position[0]), int(position[1])

    # Convert to center coordinates (PathGen uses centers as primary coordinate system)
    half_patch = snakemake.params.patch_size // 2
    x_center = x_topleft + half_patch
    y_center = y_topleft + half_patch

    # Validate position is within WSI bounds (using top-left for bounds checking)
    if (
        x_topleft + snakemake.params.patch_size > wsi_width
        or y_topleft + snakemake.params.patch_size > wsi_height
        or x_topleft < 0
        or y_topleft < 0
    ):
        logging.warning(
            f"Position {x_topleft},{y_topleft} out of bounds for WSI {file_id} (size: {wsi_width}x{wsi_height})"
        )
        continue

    # Use center coordinates for patch ID and storage
    patch_id = f"{entry['wsi_id']}_x{x_center}_y{y_center}"
    obs_data.append(
        {
            "patch_id": patch_id,
            "x_pixel": x_center,  # Center coordinates (primary system)
            "y_pixel": y_center,  # Center coordinates (primary system)
            "natural_language_annotation": entry["caption"],
            "wsi_id": entry["wsi_id"],
            "file_id": file_id,
        }
    )
    valid_entries += 1

logging.info(f"WSI {file_id}: {valid_entries}/{len(entries)} patches are valid")

# Convert to DataFrame
obs_df = pd.DataFrame(obs_data)
if not obs_df.empty:
    obs_df.index = obs_df["patch_id"]

# Create empty var and X for h5ad structure
var_data = pd.DataFrame(index=[])
X = np.empty((len(obs_df), 0))

# Create AnnData object
adata = ad.AnnData(X=X, obs=obs_df, var=var_data)

# Store metadata in uns
adata.uns["image_width"] = wsi_width
adata.uns["image_height"] = wsi_height
adata.uns["patch_size"] = snakemake.params.patch_size
adata.uns["spot_diameter_fullres"] = snakemake.params.patch_size
adata.uns["dataset"] = "pathgen"
adata.uns["modality"] = "image_text"
adata.uns["image_path"] = str(wsi_path)
adata.uns["image_fn_stem"] = wsi_path.stem
adata.uns["file_id"] = file_id
adata.uns["coordinate_system"] = "center"  # Document that we use center coordinates
adata.uns["pixel_size"] = 0.5  # We want 20x magnification

# Ensure output directory exists
Path(snakemake.output.h5ad).parent.mkdir(parents=True, exist_ok=True)

# Save h5ad file
adata.write_h5ad(snakemake.output.h5ad)
logging.info(f"Saved {len(obs_df)} patches to {snakemake.output.h5ad}")

wsi.close()
logging.info(f"Successfully created h5ad for WSI {file_id}")