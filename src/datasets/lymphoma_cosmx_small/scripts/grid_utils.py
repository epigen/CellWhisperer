"""Grid generation and filtering utilities for lymphoma CosMx data processing."""

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def create_grid_coordinates(image_width, image_height, spot_diameter_pixels):
    """Create a grid of coordinates over the image."""
    x_coords = np.arange(0, image_width, spot_diameter_pixels)
    y_coords = np.arange(0, image_height, spot_diameter_pixels)

    grid_coords = []
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            grid_coords.append((i, j, x, y))

    coord_df = pd.DataFrame(
        grid_coords, columns=["x_array", "y_array", "x_pixel", "y_pixel"]
    )
    coord_df.index = [f"tile_{k}" for k in range(len(coord_df))]
    return coord_df


def filter_background_tiles(
    coord_df, slide_wrapper, spot_diameter_pixels, white_cutoff, crop_tile_func
):
    """Filter out white/background tiles."""
    is_white = []
    logger.info("Filtering background tiles...")

    for _, row in tqdm(coord_df.iterrows(), total=len(coord_df)):
        x = int(row.x_pixel)
        y = int(row.y_pixel)

        tile = crop_tile_func(slide_wrapper, x, y, spot_diameter_pixels)
        tile = tile[:, :, :3]  # Drop alpha if present
        whiteness = np.mean(tile)
        is_white.append(whiteness > white_cutoff)

    coord_df["is_white"] = is_white
    filtered_coords = coord_df[~coord_df["is_white"]].copy()
    logger.info(f"Kept {len(filtered_coords)} tiles out of {len(coord_df)}.")
    return filtered_coords


def aggregate_expression_data(adata_hr, filtered_coords, spot_diameter_pixels):
    """Aggregate expression data for each tile."""
    aggregated_counts = []
    cell_counts_per_tile = []
    fov_info = []
    core_info = []

    # Use the converted spatial coordinates
    hr_spatial_coords = adata_hr.obsm["spatial"]

    logger.info("Aggregating expression data onto grid...")
    for _, row in tqdm(filtered_coords.iterrows(), total=len(filtered_coords)):
        x_min = row.x_pixel
        y_min = row.y_pixel
        x_max = x_min + spot_diameter_pixels
        y_max = y_min + spot_diameter_pixels

        # Find cells within the tile
        cells_in_tile_mask = (
            (hr_spatial_coords[:, 0] >= x_min)
            & (hr_spatial_coords[:, 0] < x_max)
            & (hr_spatial_coords[:, 1] >= y_min)
            & (hr_spatial_coords[:, 1] < y_max)
        )

        cell_indices = np.where(cells_in_tile_mask)[0]
        cell_counts_per_tile.append(len(cell_indices))

        if len(cell_indices) > 0:
            # Aggregate counts (summing)
            tile_counts = adata_hr.X[cell_indices, :].sum(axis=0)
            # Handle both sparse and dense matrices
            if hasattr(tile_counts, "A1"):  # sparse matrix
                aggregated_counts.append(tile_counts.A1)
            else:  # dense matrix
                aggregated_counts.append(np.asarray(tile_counts).flatten())

            # Aggregate fov and core information
            cells_obs = adata_hr.obs.iloc[cell_indices]

            # Get unique fov values and join with comma
            if "fov" in cells_obs.columns:
                unique_fovs = cells_obs["fov"].dropna().unique()
                fov_info.append(",".join(map(str, sorted(unique_fovs))))
            else:
                fov_info.append("")

            # Get unique core values and join with comma
            if "core" in cells_obs.columns:
                unique_cores = cells_obs["core"].dropna().unique()
                core_info.append(",".join(map(str, sorted(unique_cores))))
            else:
                core_info.append("")
        else:
            # No cells in tile, append empty strings
            aggregated_counts.append(np.zeros(adata_hr.n_vars))
            fov_info.append("")
            core_info.append("")

    return aggregated_counts, cell_counts_per_tile, fov_info, core_info
