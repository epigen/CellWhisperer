#!/usr/bin/env python3
"""
Convert HEST benchmark datasets to SpotWhisperer dataset format.

This script converts the HEST benchmark format (H5 image patches + H5AD expression files)
into the standard SpotWhisperer dataset format (single H5AD files per sample with embedded images).
"""

import os
import json
import h5py
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import warnings
import logging
import anndata


# Suppress scanpy warnings
warnings.filterwarnings("ignore", category=UserWarning)
sc.settings.verbosity = 0

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_adata(expr_path, barcodes=None, normalize=False):
    """
    Load expression data from .h5ad file
    Copied from HEST processing functions
    """
    adata = sc.read_h5ad(expr_path)
    if barcodes is not None:
        adata = adata[barcodes]

    # Normalize if needed
    if normalize:
        # Log1p normalization
        filtered_adata = adata.copy()
        filtered_adata.X = filtered_adata.X.astype(np.float64)
        sc.pp.log1p(filtered_adata)
        return filtered_adata

    return adata


def reconstruct_image_from_patches(patches_h5_path, target_image_size=None):
    """
    Reconstruct a full image from HEST patches stored in H5 file.

    This creates a pseudo-WSI by arranging patches based on their coordinates.
    Not perfect but allows us to create the expected format.
    """
    with h5py.File(patches_h5_path, "r") as f:
        imgs = f["img"][:]
        coords = f["coords"][:]  # Should be (x, y) coordinates

    logger.info(
        f"Reconstructing image from {len(imgs)} patches of size {imgs[0].shape}"
    )

    # Get patch size
    patch_h, patch_w = imgs[0].shape[:2]
    print(patch_h)

    # Determine image bounds from coordinates
    min_x, min_y = coords.min(axis=0).astype(int)
    max_x, max_y = coords.max(axis=0).astype(int)
    if min_x < 112:
        print("min_x < 112")
    min_x = 0
    min_y = 0

    # Create canvas without padding
    canvas_w = max_x - min_x + patch_w
    canvas_h = max_y - min_y + patch_h

    if target_image_size is not None:
        # Resize canvas if specified
        canvas_w = min(canvas_w, target_image_size)
        canvas_h = min(canvas_h, target_image_size)

    # Create empty canvas
    reconstructed_image = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place patches on canvas
    for i, (patch, (x, y)) in enumerate(zip(imgs, coords)):
        # Adjust coordinates relative to canvas
        canvas_x = int(x - min_x)
        canvas_y = int(y - min_y)

        # Ensure patch fits in canvas
        assert (
            canvas_x + patch_w <= canvas_w
            and canvas_y + patch_h <= canvas_h
            and canvas_x >= 0
            and canvas_y >= 0
        ), f"Patch {i} at {(canvas_x, canvas_y)} does not fit in canvas {(canvas_w, canvas_h)}"

        reconstructed_image[
            canvas_y : canvas_y + patch_h, canvas_x : canvas_x + patch_w
        ] = patch

    logger.info(f"Reconstructed image shape: {reconstructed_image.shape}")
    return reconstructed_image, (
        patch_w // 2,
        patch_h // 2,
    )  # Return offset for coordinate adjustment


def convert_hest_sample_to_spotwhisperer(sample_row, dataset_bench_path, output_path):
    """
    Convert a single HEST sample to SpotWhisperer format.

    Args:
        sample_row: Row from HEST split CSV containing sample info
        dataset_bench_path: Path to HEST dataset
        output_path: Path to save converted sample
    """
    sample_id = sample_row["sample_id"]
    patches_path = sample_row["patches_path"]
    expr_path = sample_row["expr_path"]

    logger.info(f"Converting sample {sample_id}")

    # Load patches data
    patches_h5_path = dataset_bench_path / patches_path
    expr_h5ad_path = dataset_bench_path / expr_path

    # Load patches and reconstruct image
    with h5py.File(patches_h5_path, "r") as f:
        barcodes = f["barcode"][:].flatten().astype(str).tolist()
        coords = f["coords"][:]  # (n_spots, 2) - pixel coordinates

    # Reconstruct image from patches
    reconstructed_image, coord_offset = reconstruct_image_from_patches(patches_h5_path)

    # Load expression data
    adata = load_adata(str(expr_h5ad_path), barcodes=barcodes, normalize=False)

    # Adjust coordinates relative to reconstructed image
    adjusted_coords = coords + np.array(coord_offset)

    # Create observation data
    obs_data = pd.DataFrame(
        {
            "patch_id": [f"{sample_id}_{i:04d}" for i in range(len(barcodes))],
            "x_pixel": adjusted_coords[:, 0].astype(int),
            "y_pixel": adjusted_coords[:, 1].astype(int),
            "barcode": barcodes,
        }
    )
    obs_data.index = obs_data["patch_id"]

    # Create new AnnData object in SpotWhisperer format
    new_adata = ad.AnnData(
        X=adata.X,
        obs=obs_data,
        var=adata.var.copy(),
    )

    # Copy layers if they exist
    for layer_name in adata.layers.keys():
        new_adata.layers[layer_name] = adata.layers[layer_name]

    # Add required uns fields for SpotWhisperer
    new_adata.uns["he_slide"] = reconstructed_image
    new_adata.uns["spot_diameter_fullres"] = snakemake.params["patch_size_pixels"]
    new_adata.uns["pixel_size"] = 0.25
    new_adata.uns["dataset"] = "hest_benchmark"
    new_adata.uns["modality"] = "spatial_transcriptomics"
    new_adata.uns["sample_id"] = sample_id
    new_adata.uns["image_width"] = reconstructed_image.shape[1]
    new_adata.uns["image_height"] = reconstructed_image.shape[0]
    new_adata.uns["patch_size"] = snakemake.params["patch_size_pixels"]

    # Add spatial information (required by some downstream tools)
    new_adata.obsm["spatial"] = adjusted_coords

    # Save in SpotWhisperer format
    new_adata.write_h5ad(output_path)

    logger.info(f"Saved converted sample to {output_path}")
    logger.info(f"Sample contains {new_adata.n_obs} spots and {new_adata.n_vars} genes")


def create_dataset_splits(all_converted_files, train_samples, test_samples, output_dir):
    """
    Create train/test splits for the converted dataset.
    """
    # Create splits directory
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    # Map sample IDs to converted file paths
    sample_to_file = {}
    for file_path in all_converted_files:
        sample_id = Path(file_path).stem
        sample_to_file[sample_id] = file_path

    # Create train split
    train_files = []
    for sample_id in train_samples:
        if sample_id in sample_to_file:
            train_files.append(sample_to_file[sample_id])

    # Create test split
    test_files = []
    for sample_id in test_samples:
        if sample_id in sample_to_file:
            test_files.append(sample_to_file[sample_id])

    # Save splits
    train_df = pd.DataFrame({"file_path": train_files})
    test_df = pd.DataFrame({"file_path": test_files})

    train_df.to_csv(splits_dir / "train.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    logger.info(
        f"Created splits: {len(train_files)} train, {len(test_files)} test samples"
    )

    return splits_dir


# Get paths from Snakemake
dataset_bench_path = Path(snakemake.input.dataset_dir)
output_dir = Path(snakemake.params.multi_folder)
output_dir.mkdir(parents=True, exist_ok=True)

logger.info(f"Converting HEST dataset from {dataset_bench_path}")


# Find all split files to get sample information
splits_dir = dataset_bench_path / "splits"

# Get all unique samples from all splits

dfs = []

for split_file in splits_dir.glob("*.csv"):  # TODO optionally filter by test
    split_df = pd.read_csv(split_file)
    dfs.append(split_df)

# unique
df = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
df["relative_path"] = df["sample_id"].apply(
    lambda sample_id: (
        output_dir.relative_to(Path(snakemake.output.converted_dataset).parent)
        / f"full_data_{sample_id}.h5ad"
    ).as_posix()
)

# Convert each unique sample

for _, sample_row in tqdm(df.iterrows(), total=len(df)):
    convert_hest_sample_to_spotwhisperer(
        sample_row,
        dataset_bench_path,
        output_path=Path(snakemake.output.converted_dataset).parent
        / sample_row["relative_path"],
    )


# Create dataset metadata
metadata = {
    "dataset_name": f"hesteval_{snakemake.wildcards.dataset}",
    "modality": "spatial_transcriptomics",
    "multi_sample_fns": df["relative_path"].tolist(),
    "multi_sample_ids": df["sample_id"].tolist(),
    "patch_size": snakemake.params["patch_size_pixels"],
    "converted_from": "hest_benchmark",
}
adata = anndata.AnnData()
adata.uns = metadata

adata.write_h5ad(snakemake.output.converted_dataset)

logger.info("Dataset conversion completed successfully")
