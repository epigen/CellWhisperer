"""
Process lymphoma CosMx data into gridded or single-cell formats.

Core purpose: Process an image and h5ad file into spatially resolved outputs.
"""

import anndata
import logging
import numpy as np
import pandas as pd
from PIL import Image

from coordinate_utils import (
    convert_mm_to_pixel_coordinates,
    perform_core_level_alignment,
)
from image_utils import (
    OpenSlideWrapper,
    handle_magnification,
    crop_tile,
    prepare_visualization_image,
)
from visualization import save_example_patches


def process_singlecell_data(adata_hr, spatial_coords, spot_diameter_pixels):
    """Process single-cell data by creating coordinate dataframe and output AnnData."""
    logging.info("Single cell mode: using spatial coordinates from h5ad file")

    # Extract pixel coordinates from the spatial obsm data
    pixel_coords = adata_hr.obsm["spatial"]

    # Create filtered_coords DataFrame with x_pixel and y_pixel from spatial coordinates
    filtered_coords = pd.DataFrame(
        {"x_pixel": pixel_coords[:, 0], "y_pixel": pixel_coords[:, 1]}
    )
    filtered_coords.index = adata_hr.obs_names

    # Add array grid coordinates
    # filtered_coords["x_array"] = filtered_coords["x_pixel"] // spot_diameter_pixels  # NOTE not necessary?
    # filtered_coords["y_array"] = filtered_coords["y_pixel"] // spot_diameter_pixels

    # Create output AnnData using original cell data
    logging.info(
        "Single cell mode: using individual cell transcriptomes (no aggregation)"
    )
    adata = adata_hr  # .copy()

    # Add coordinate information from filtered_coords to match cell order
    # adata.obs["x_array"] = filtered_coords["x_array"]
    # adata.obs["y_array"] = filtered_coords["y_array"]
    adata.obs["x_pixel"] = filtered_coords["x_pixel"]
    adata.obs["y_pixel"] = filtered_coords["y_pixel"]
    adata.obs["n_cells"] = 1  # Each "tile" contains exactly 1 cell
    adata.obsm["spatial"] = spatial_coords

    return adata, filtered_coords


logging.basicConfig(level=logging.INFO)

# Get parameters from snakemake
dataset = snakemake.wildcards.dataset
sample = snakemake.wildcards.sample
SPOT_DIAMETER_PIXELS = snakemake.params.spot_diameter_pixels
WHITE_CUTOFF = snakemake.params.white_cutoff
BRIGHTNESS_THRESHOLD = snakemake.params.brightness_threshold
X_OFFSET_PIXELS = snakemake.params.x_offset_pixels
Y_OFFSET_PIXELS = snakemake.params.y_offset_pixels
SCALE_FACTOR_X = snakemake.params.scale_factor_x
SCALE_FACTOR_Y = snakemake.params.scale_factor_y
logging.info(
    f"Processing {dataset} sample {sample} with spot_diameter_pixels={SPOT_DIAMETER_PIXELS}, "
    f"white_cutoff={WHITE_CUTOFF}, brightness_threshold={BRIGHTNESS_THRESHOLD}, "
    f"x_offset_pixels={X_OFFSET_PIXELS}, y_offset_pixels={Y_OFFSET_PIXELS}, "
    f"scale_factor_x={SCALE_FACTOR_X}, scale_factor_y={SCALE_FACTOR_Y}, "
    f"pixel_size={snakemake.params.pixel_size_um} μm/pixel"
)


# Load data
logging.info(f"Loading AnnData from {snakemake.input.read_count_table}")
adata_hr = anndata.read_h5ad(snakemake.input.read_count_table, backed="r")

# Map existing coreID field to core_id for alignment function
adata_hr.obs["core_id"] = adata_hr.obs["TMA_coreID"]

# Record pixel size metadata before any magnification adjustments
adata_hr.uns["pixel_size"] = snakemake.params.pixel_size_um

# Load image at full resolution
logging.info(f"Loading image from {snakemake.input.image}")
slide_wrapper = OpenSlideWrapper(snakemake.input.image)
slide_wrapper = handle_magnification(adata_hr, slide_wrapper)

# Get actual image dimensions
actual_width, actual_height = slide_wrapper.size
logging.info(f"Actual image dimensions: {actual_width}x{actual_height}")

# Use fixed reference dimensions for coordinate scaling # This is just a constant for more reasonable scaling factors
full_img_width = 49800
full_img_height = 49800
logging.info(
    f"Using fixed reference dimensions for coordinate scaling: {full_img_width}x{full_img_height}"
)

# Convert coordinates from mm to pixels
spatial_coords = convert_mm_to_pixel_coordinates(
    adata_hr,
    full_img_width,
    full_img_height,
    X_OFFSET_PIXELS,
    Y_OFFSET_PIXELS,
    SCALE_FACTOR_X,
    SCALE_FACTOR_Y,
)

# Apply core-level alignment if requested
logging.info("Applying core-level alignment optimization")
aligned_coords, cell_intensities = perform_core_level_alignment(
    adata_hr, slide_wrapper, spatial_coords, dataset
)
adata_hr.obsm["spatial"] = aligned_coords

# Add brightness-based quality columns
adata_hr.obs["cell_brightness"] = cell_intensities
adata_hr.obs["is_too_bright"] = cell_intensities > BRIGHTNESS_THRESHOLD
logging.info(
    f"Flagged {adata_hr.obs['is_too_bright'].sum()} cells as too bright "
    f"(>{BRIGHTNESS_THRESHOLD}) out of {len(adata_hr)} total cells "
    f"({adata_hr.obs['is_too_bright'].mean()*100:.1f}%)"
)

# Process single-cell data
adata, filtered_coords = process_singlecell_data(
    adata_hr, adata_hr.obsm["spatial"], SPOT_DIAMETER_PIXELS
)

# Prepare visualization image and add spatial metadata
img_np, scale_factor_img = prepare_visualization_image(slide_wrapper)

library_id = f"{dataset}_{sample}"
adata.uns["spatial"] = {
    library_id: {
        "images": {"hires": img_np},
        "scalefactors": {
            "tissue_hires_scalef": scale_factor_img,
            "spot_diameter_fullres": SPOT_DIAMETER_PIXELS,
        },
        "metadata": {"library_id": library_id},
    }
}

# Add image_path and pixel_size to uns
adata.uns["image_path"] = str(snakemake.input.image)
adata.uns["pixel_size"] = adata_hr.uns["pixel_size"]
adata.uns["dataset"] = dataset
adata.uns["sample"] = sample

# Generate outputs
if hasattr(snakemake.output, "qc_tile_plot"):
    Image.fromarray(img_np.astype(np.uint8)).save(snakemake.output.qc_tile_plot)

if (
    hasattr(snakemake.output, "report_patches")
    and len(snakemake.output.report_patches) > 0
):
    save_example_patches(
        filtered_coords,
        slide_wrapper,
        SPOT_DIAMETER_PIXELS,
        snakemake.output.report_patches,
        crop_tile,
        snakemake.params.pixel_size_um,
        snakemake.params.small_patch_diameter_um,
    )

# Close the slide if it was opened
if slide_wrapper is not None:
    slide_wrapper.close()

# Save processed data
logging.info(f"Saving gridded AnnData to {snakemake.output.adata}")
adata.write_h5ad(snakemake.output.adata)
logging.info("Processing complete.")
