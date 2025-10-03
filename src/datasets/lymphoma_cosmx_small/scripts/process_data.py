"""
Process lymphoma CosMx data into gridded format.

Core purpose: Process an image and h5ad file into patch region coordinates,
for which read counts are aggregated.
"""

import anndata
import logging
import shutil
import numpy as np

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
SPOT_DIAMETER_PIXELS = snakemake.params.spot_diameter_pixels
WHITE_CUTOFF = snakemake.params.white_cutoff
X_OFFSET = snakemake.params.x_offset
Y_OFFSET = snakemake.params.y_offset
SCALE_FACTOR = snakemake.params.scale_factor
FILTER_GOOD_QUALITY = snakemake.params.filter_good_quality

logging.info(
    f"Processing {dataset} with spot_diameter_pixels={SPOT_DIAMETER_PIXELS}, "
    f"white_cutoff={WHITE_CUTOFF}, x_offset={X_OFFSET}, y_offset={Y_OFFSET}, scale_factor={SCALE_FACTOR}, "
    f"filter_good_quality={FILTER_GOOD_QUALITY}"
)


# Load data
logging.info(f"Loading AnnData from {snakemake.input.read_count_table}")
adata_hr = anndata.read_h5ad(snakemake.input.read_count_table)

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

# Create and filter grid
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

gridded_adata = anndata.AnnData(counts_matrix, var=adata_hr.var)
gridded_adata.obs.index = filtered_coords.index
gridded_adata.obs["x_array"] = filtered_coords["x_array"]
gridded_adata.obs["y_array"] = filtered_coords["y_array"]
gridded_adata.obs["x_pixel"] = filtered_coords["x_pixel"]
gridded_adata.obs["y_pixel"] = filtered_coords["y_pixel"]
gridded_adata.obs["n_cells"] = cell_counts_per_tile
gridded_adata.obs["fov_info"] = fov_info
gridded_adata.obs["core_info"] = core_info
gridded_adata.obsm["spatial"] = filtered_coords[["x_pixel", "y_pixel"]].values

# Filter for min number of cells per tile
gridded_adata = gridded_adata[
    gridded_adata.obs["n_cells"] >= snakemake.params["min_num_cells"]
].copy()

# Prepare visualization image and add spatial metadata
img_np, scale_factor_img = prepare_visualization_image(slide_wrapper)
library_id = "lymphoma_cosmx_small"
gridded_adata.uns["spatial"] = {
    library_id: {
        "images": {"hires": img_np},
        "scalefactors": {
            "tissue_hires_scalef": scale_factor_img,
            "spot_diameter_fullres": SPOT_DIAMETER_PIXELS,
        },
        "metadata": {"library_id": library_id},
    }
}

# Add image_path to uns and copy the SVS file
gridded_adata.uns["image_path"] = str(snakemake.output.image)


# Copy the original SVS file to the output location
logging.info(
    f"Copying SVS file from {snakemake.input.image} to {snakemake.output.image}"
)

shutil.copy2(snakemake.input.image, snakemake.output.image)

# Generate outputs
if hasattr(snakemake.output, "qc_tile_plot"):
    generate_qc_tile_plot(
        coord_df,
        img_np,
        SPOT_DIAMETER_PIXELS,
        scale_factor_img,
        snakemake.output.qc_tile_plot,
    )

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
gridded_adata.write_h5ad(snakemake.output.adata)
logging.info("Processing complete.")
