import shutil
import gzip
from pathlib import Path
import pandas as pd
import scanpy as sc
import numpy as np
import math
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCALE_FACTOR = 500
VIEW_X = 500  # approx. viewport width


def augment_metadata(adata):
    """
    Augment the SpotWhisperer dataset metadata to be compatible with CellWhisperer.
    Based on the dataset description in the issue.
    """
    # Use raw_counts layer if available, otherwise DeepSpot-inferred expression
    if "raw_counts" in adata.layers:  # TODO delete
        adata.layers["counts"] = adata.layers["raw_counts"].astype(int)
    else:
        # If raw_counts not available, use the main matrix (DeepSpot-inferred)
        adata.layers["counts"] = adata.X.astype(int)

    # Ensure gene names are available
    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var.index

    # Handle spatial information - check for different coordinate systems
    if "spatial" in adata.obsm:
        # Use existing spatial coordinates
        adata.obsm["X_spatial"] = adata.obsm["spatial"]
    elif "x_pixel" in adata.obs.columns and "y_pixel" in adata.obs.columns:
        # Use pixel coordinates
        adata.obsm["X_spatial"] = pd.concat(
            (adata.obs["x_pixel"], adata.obs["y_pixel"]), axis=1
        ).to_numpy()
    elif "x_array" in adata.obs.columns and "y_array" in adata.obs.columns:
        # Use array coordinates as fallback
        adata.obsm["X_spatial"] = pd.concat(
            (adata.obs["x_array"], adata.obs["y_array"]), axis=1
        ).to_numpy()

    adata.uns["default_embedding"] = "X_spatial"
    adata.uns["dataset"] = "lung_tissue"

    return adata  # it's anyways inplace


def process_single_file(file_path, output_path):
    """Process a single h5ad.gz file."""
    logger.info(f"Processing {file_path}")

    # Read the compressed file
    adata = sc.read_h5ad(file_path)

    if "he_slide" in adata.uns:  # TODO temporary fix for compatibility
        adata.uns["20x_slide"] = adata.uns["he_slide"]

    # Add sample information
    sample_name = Path(file_path).stem.replace(".h5ad", "")
    adata.obs["sample_id"] = sample_name

    # Apply augmentation
    adata = augment_metadata(adata)

    # Save processed file
    adata.write_h5ad(output_path)
    logger.info(f"Saved processed file to {output_path}")

    return adata


def combine_files(file_paths, output_path):
    """Combine multiple h5ad files into one."""
    logger.info(f"Combining {len(file_paths)} files")

    adatas = []
    wsi_images = {}  # Store WSI images keyed by sample_id

    for file_path in file_paths:
        adata = sc.read_h5ad(file_path)
        # Add sample information to each dataset
        sample_name = Path(file_path).stem.replace(".h5ad", "")
        adata.obs["sample_id"] = sample_name

        if "he_slide" in adata.uns:  # TODO temporary fix for compatibility
            adata.uns["20x_slide"] = adata.uns["he_slide"]

        # Check if this file has a whole slide image
        if "20x_slide" in adata.uns:
            wsi_images[sample_name] = adata.uns["20x_slide"]
            logger.info(f"Found WSI image in {sample_name}")

        adatas.append(adata)

    # Concatenate all datasets
    combined_adata = sc.concat(adatas, merge="same")
    combined_adata = augment_metadata(combined_adata)

    # Store all WSI images in the combined dataset's uns
    if wsi_images:
        combined_adata.uns["wsi_images"] = wsi_images
        logger.info(f"Stored {len(wsi_images)} WSI images in combined dataset")

        # For compatibility with UniProcessor, also store the first available WSI as the default
        # This allows models to work with the first sample's WSI by default
        first_sample = list(wsi_images.keys())[0]
        combined_adata.uns["20x_slide"] = wsi_images[first_sample]
        logger.info(f"Set default WSI to {first_sample}")

    # Save combined file
    combined_adata.write_h5ad(output_path)
    logger.info(f"Saved combined file to {output_path}")

    return combined_adata


# Main execution
logger.info("Starting lung tissue dataset processing")

# Process input files (already downloaded by snakemake HTTP.remote)
input_files = snakemake.input.sample_files
logger.info(f"Processing {len(input_files)} input files")

if len(input_files) == 1:
    # Only one file, process it directly
    process_single_file(input_files[0], snakemake.output.dataset)
else:
    # Multiple files, combine them
    combine_files(input_files, snakemake.output.dataset)

logger.info("Lung tissue dataset processing completed")
