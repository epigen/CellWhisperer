"""
Process downloaded HEST-1K dataset for SpotWhisperer.

This script processes already downloaded HEST data to be compatible
with UNIProcessor requirements.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
import logging
from hest import iter_hest


def prepare_adata_for_uniprocessor(st_data):
    """
    Process HEST data to match UNIProcessor requirements.

    Args:
        st_data: HESTData object from HEST library

    Returns:
        adata: Processed AnnData object compatible with UNIProcessor
    """
    adata = st_data.adata.copy()

    # Ensure required spatial coordinates are available
    # TODO: Fix this by linking WSI to pxl_col_in_fullres coordinates properly
    if "pxl_col_in_fullres" in adata.obs and "pxl_row_in_fullres" in adata.obs:
        # Use HEST's pixel coordinates
        adata.obs["x_pixel"] = adata.obs["pxl_col_in_fullres"].astype(
            int
        )  # pxl_col_in_fullres is centered on spot (https://github.com/mahmoodlab/HEST/blob/main/tutorials/2-Interacting-with-HEST-1k.ipynb)
        adata.obs["y_pixel"] = adata.obs["pxl_row_in_fullres"].astype(int)
    elif "spatial" in adata.obsm:
        # Use spatial coordinates from obsm
        adata.obs["x_pixel"] = adata.obsm["spatial"][:, 0].astype(int)
        adata.obs["y_pixel"] = adata.obsm["spatial"][:, 1].astype(int)
    else:
        raise ValueError("No spatial coordinates found in HEST data")

    # Get WSI image from HEST data - prioritize WSI first
    # HEST provides different image access methods
    try:
        # Get image from HEST data object - WSI corresponds to pxl_col_in_fullres coordinates
        wsi = st_data.wsi

        image = (
            wsi.numpy()
        )  # np.array(wsi.read_region((0, 0), 0, wsi.level_dimensions[0]))

        # Remove alpha channel if present

        if image.shape[-1] == 4:
            image = image[:, :, :3]

    except Exception as e:
        raise ValueError(f"Could not load WSI image: {e}")

    adata.uns["20x_slide"] = image
    adata.uns["pixel_size"] = st_data.pixel_size  # um per pixel
    adata.uns["spot_diameter_fullres"] = adata.uns["spatial"]["ST"]["scalefactors"][
        "spot_diameter_fullres"
    ]

    # elif "downscaled_fullres" in adata.uns.get("spatial", {}).get("ST", {}).get(
    #     "images", {}
    # ):
    #     # If there is no WSI but a downscaled_fullres, use these coordinates
    #     image = adata.uns["spatial"]["ST"]["images"]["downscaled_fullres"]
    #     adata.uns["20x_slide"] = image
    # else:
    #     raise ValueError(
    #         "No functioning image found - neither WSI nor downscaled_fullres available"
    #     )

    # Set spot diameter for patch extraction (typical for spatial transcriptomics)
    # HEST data should have this information, but provide fallback
    # TODO: Investigate the spot diameter thing in more detail

    # Ensure gene names are available
    adata.var.index = adata.var.index.astype(str).map(str.upper)

    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var.index

    # Add counts layer if not present (required by some scFMs)
    if "counts" not in adata.layers:
        # Ensure .X is not in log-space by checking if it contains values > 10
        # If it looks like log-space, exponentiate and convert to int
        if hasattr(adata.X, "toarray"):
            x_data = adata.X.toarray()
        else:
            x_data = adata.X

        # Check if data looks like log-transformed (typically small positive values)
        if np.max(x_data) < 15 and np.min(x_data) >= 0:
            # Likely log-transformed, convert back to counts
            logging.warning(
                "Data appears to be log-transformed, converting back to counts"
            )
            adata.layers["counts"] = np.expm1(x_data).astype(int)
        else:
            # Data appears to be raw counts already
            adata.layers["counts"] = x_data.astype(int)

    return adata


sample_id = snakemake.params.sample_id

print(f"Processing sample {sample_id}")

# Use HEST's iter_hest to load and process the data
for st_data in iter_hest(str(snakemake.params.hest_cache_dir), id_list=[sample_id]):
    # Process the data for UNIProcessor compatibility
    adata = prepare_adata_for_uniprocessor(st_data)

    # Add sample ID to uns for tracking
    adata.uns["sample_id"] = sample_id
    adata.uns["dataset"] = "hest1k"

    # Save individual sample file (full_dataset_multi pattern)
    print(f"Saving processed data to {snakemake.output.full_data_file}")
    adata.write_h5ad(snakemake.output.full_data_file)

    print(f"Successfully processed sample {sample_id}")
