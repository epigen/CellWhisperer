"""
Process lymphoma CosMx data into gridded or single-cell formats.

Core purpose: Process an image and h5ad file into spatially resolved outputs.
"""

import anndata
import logging
import re
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from coordinate_utils import convert_mm_to_pixel_coordinates
from image_utils import (
    OpenSlideWrapper,
    handle_magnification,
    crop_tile,
    prepare_visualization_image,
)
from grid_utils import (
    create_grid_coordinates,
    filter_background_tiles,
    aggregate_expression_data,
)
from visualization import generate_qc_tile_plot, save_example_patches

logging.basicConfig(level=logging.INFO)

# Get parameters from snakemake
dataset = snakemake.wildcards.dataset
IS_SINGLECELL = dataset.endswith("_singlecell")
SPOT_DIAMETER_PIXELS = snakemake.params.spot_diameter_pixels
WHITE_CUTOFF = snakemake.params.white_cutoff
X_OFFSET = snakemake.params.x_offset
Y_OFFSET = snakemake.params.y_offset
SCALE_FACTOR = snakemake.params.scale_factor
FILTER_GOOD_QUALITY = snakemake.params.filter_good_quality

logging.info(
    f"Processing {dataset} with spot_diameter_pixels={SPOT_DIAMETER_PIXELS}, "
    f"white_cutoff={WHITE_CUTOFF}, x_offset={X_OFFSET}, y_offset={Y_OFFSET}, scale_factor={SCALE_FACTOR}, "
    f"filter_good_quality={FILTER_GOOD_QUALITY}, pixel_size={snakemake.params.pixel_size_um} μm/pixel"
)


# Load data
logging.info(f"Loading AnnData from {snakemake.input.read_count_table}")
adata_hr = anndata.read_h5ad(snakemake.input.read_count_table)

# Map cell barcodes to core IDs (optional)
core_assignment_path = getattr(snakemake.input, "cell_barcode_core_assignment", None)
if core_assignment_path:
    logging.info(f"Loading cell barcode to core mapping from {core_assignment_path}")
    core_df = pd.read_csv(core_assignment_path)
    required_cols = {"cell_barcode", "core_id"}
    missing_cols = required_cols - set(core_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols} in {core_assignment_path}")
    barcode_to_core = dict(
        zip(core_df["cell_barcode"].astype(str), core_df["core_id"].astype(str))
    )
    adata_hr.obs["core_id"] = adata_hr.obs_names.astype(str).map(barcode_to_core)
    missing_core_ids = adata_hr.obs["core_id"].isna().sum()
    if missing_core_ids > 0:
        logging.warning(
            f"{missing_core_ids} cells did not have a matching core assignment"
        )
else:
    logging.warning(
        "No cell barcode to core mapping provided; core_id will remain empty"
    )

# Filter for good quality cells if requested
if FILTER_GOOD_QUALITY and "good_quality" in adata_hr.obs.columns:
    n_cells_before = adata_hr.n_obs
    adata_hr = adata_hr[adata_hr.obs["good_quality"]].copy()
    n_cells_after = adata_hr.n_obs
    logging.info(
        f"Filtered for good_quality cells: {n_cells_before} -> {n_cells_after} cells"
    )
elif FILTER_GOOD_QUALITY:
    logging.warning(
        "good_quality column not found in adata.obs, skipping quality filtering"
    )

# Record pixel size metadata before any magnification adjustments
adata_hr.uns["pixel_size"] = snakemake.params.pixel_size_um

logging.info(f"Loading image from {snakemake.input.image}")
slide_wrapper = OpenSlideWrapper(snakemake.input.image, dataset)

# Handle magnification scaling
original_width, original_height = slide_wrapper.slide.level_dimensions[
    0
]  # Level 0 dimensions
slide_wrapper = handle_magnification(adata_hr, slide_wrapper, dataset)

# Convert coordinates from mm to pixels
width, height = slide_wrapper.size
image_was_scaled = (width != original_width) or (height != original_height)

if image_was_scaled:
    logging.info("Image was scaled - adjusting coordinate transformation")
    spatial_coords = convert_mm_to_pixel_coordinates(
        adata_hr, original_width, original_height, X_OFFSET, Y_OFFSET, SCALE_FACTOR
    )
    # Scale coordinates to match the scaled image
    scale_factor_x = width / original_width
    scale_factor_y = height / original_height
    spatial_coords[:, 0] *= scale_factor_x
    spatial_coords[:, 1] *= scale_factor_y
    logging.info(
        f"Scaled coordinates by factors: x={scale_factor_x:.3f}, y={scale_factor_y:.3f}"
    )
else:
    logging.info("Image was not scaled - using direct coordinate transformation")
    spatial_coords = convert_mm_to_pixel_coordinates(
        adata_hr, width, height, X_OFFSET, Y_OFFSET, SCALE_FACTOR
    )

adata_hr.obsm["spatial"] = spatial_coords

# Create and filter grid or use CSV coordinates for single cell
if IS_SINGLECELL:
    # Read CSV coordinates from input
    logging.info("Single cell mode: reading coordinates from CSV")
    csv_coords = pd.read_csv(
        snakemake.input.read_count_table.replace(".h5ad", "_coordinates.csv")
    )

    # Create filtered_coords DataFrame with x_pixel and y_pixel from CSV
    filtered_coords = pd.DataFrame(
        {"x_pixel": csv_coords["x_pixel"], "y_pixel": csv_coords["y_pixel"]}
    )
    filtered_coords.index = (
        csv_coords.index if "index" not in csv_coords.columns else csv_coords["index"]
    )

    # Add missing columns to match grid format
    filtered_coords["x_array"] = filtered_coords["x_pixel"] // SPOT_DIAMETER_PIXELS
    filtered_coords["y_array"] = filtered_coords["y_pixel"] // SPOT_DIAMETER_PIXELS
    coord_df = None  # No coord_df for single cell mode
else:
    coord_df = create_grid_coordinates(width, height, SPOT_DIAMETER_PIXELS)
    filtered_coords = filter_background_tiles(
        coord_df, slide_wrapper, SPOT_DIAMETER_PIXELS, WHITE_CUTOFF, crop_tile
    )

# Aggregate expression data
aggregated_counts, cell_counts_per_tile, fov_info, core_info = (
    aggregate_expression_data(adata_hr, filtered_coords, SPOT_DIAMETER_PIXELS)
)

# Create gridded AnnData object
if aggregated_counts:
    counts_matrix = np.vstack(aggregated_counts)
else:
    counts_matrix = np.empty((0, adata_hr.n_vars))

adata = anndata.AnnData(counts_matrix, var=adata_hr.var)
adata.obs.index = filtered_coords.index
adata.obs["x_array"] = filtered_coords["x_array"]
adata.obs["y_array"] = filtered_coords["y_array"]
adata.obs["x_pixel"] = filtered_coords["x_pixel"]
adata.obs["y_pixel"] = filtered_coords["y_pixel"]
adata.obs["n_cells"] = cell_counts_per_tile
adata.obs["fov_info"] = fov_info
adata.obs["core_info"] = core_info
adata.obsm["spatial"] = filtered_coords[["x_pixel", "y_pixel"]].values

# Filter for min number of cells per tile
adata = adata[adata.obs["n_cells"] >= snakemake.params["min_num_cells"]].copy()

# Prepare visualization image and add spatial metadata
img_np, scale_factor_img = prepare_visualization_image(slide_wrapper)
library_id = "lymphoma_cosmx_small"
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

# Add image_path and pixel_size to uns and copy the SVS file
adata.uns["image_path"] = str(snakemake.output.image)
adata.uns["pixel_size"] = adata_hr.uns["pixel_size"]


# Copy the original SVS file to the output location
logging.info(
    f"Copying SVS file from {snakemake.input.image} to {snakemake.output.image}"
)

shutil.copy2(snakemake.input.image, snakemake.output.image)

# Generate outputs
if hasattr(snakemake.output, "qc_tile_plot"):
    if not IS_SINGLECELL:
        generate_qc_tile_plot(
            coord_df,
            img_np,
            SPOT_DIAMETER_PIXELS,
            scale_factor_img,
            snakemake.output.qc_tile_plot,
        )
    else:
        # For single cell, just save the image
        from PIL import Image

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
    )

# Close the slide
slide_wrapper.close()

# Save processed data
logging.info(f"Saving gridded AnnData to {snakemake.output.adata}")
adata.write_h5ad(snakemake.output.adata)
logging.info("Processing complete.")
