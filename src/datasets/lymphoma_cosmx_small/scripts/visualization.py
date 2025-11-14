"""Visualization utilities for lymphoma CosMx data processing."""

"copy, outdated, use notebooks/visualization.py instead"

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def generate_qc_tile_plot(
    coord_df, img_np, spot_diameter_pixels, scale_factor_img, output_path
):
    """Generate QC plot showing tile selection using center-anchored coordinates.

    Assumes `coord_df` provides center coordinates (`x_pixel`, `y_pixel`).
    """
    logger.info("Generating QC tile plot (center-anchored)...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_np)
    scaled_spot_diameter = spot_diameter_pixels * scale_factor_img
    half = scaled_spot_diameter / 2.0

    for _, row in coord_df.iterrows():
        x_center = row.x_pixel * scale_factor_img
        y_center = row.y_pixel * scale_factor_img
        # Draw tile rectangle centered at (x_center, y_center)
        rect_x = x_center - half
        rect_y = y_center - half
        color = "red" if getattr(row, "is_white", False) else "blue"
        rect = patches.Rectangle(
            (rect_x, rect_y),
            scaled_spot_diameter,
            scaled_spot_diameter,
            linewidth=0.5,
            edgecolor=color,
            facecolor="none",
            alpha=0.5,
        )
        ax.add_patch(rect)

        # Draw UNI cell view (56x56) as a red rectangle centered at the same coordinates
        scaled_cell_diameter = 56 * scale_factor_img
        cell_half = scaled_cell_diameter / 2.0
        cell_rect = patches.Rectangle(
            (x_center - cell_half, y_center - cell_half),
            scaled_cell_diameter,
            scaled_cell_diameter,
            linewidth=0.7,
            edgecolor="red",
            facecolor="none",
            alpha=0.7,
        )
        ax.add_patch(cell_rect)
    ax.set_title("Tile Selection (Blue=Kept, Red=Discarded)")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved QC tile plot to {output_path}")


def save_example_patches(
    filtered_coords, slide_wrapper, spot_diameter_pixels, output_paths, crop_tile_func
):
    """Save random example patches for report using center-anchored coordinates.

    Assumes `filtered_coords` provides center coordinates (`x_pixel`, `y_pixel`).
    """
    if len(output_paths) == 0:
        return

    logger.info(f"Saving {len(output_paths)} random test patches (center-anchored).")
    # Ensure we don't try to sample more tiles than available
    num_patches_to_save = min(len(filtered_coords), len(output_paths))

    if num_patches_to_save > 0:
        # Randomly sample tiles
        sample_tiles = filtered_coords.sample(n=num_patches_to_save, random_state=42)

        for i, (index, row) in enumerate(sample_tiles.iterrows()):
            x_center = int(row.x_pixel)
            y_center = int(row.y_pixel)

            # Convert center to top-left for the crop function
            x_start = x_center - (spot_diameter_pixels // 2)
            y_start = y_center - (spot_diameter_pixels // 2)

            tile_array = crop_tile_func(slide_wrapper, x_start, y_start, spot_diameter_pixels)
            # Ensure RGB (drop alpha if present)
            tile_array = tile_array[:, :, :3]
            patch_image = Image.fromarray(tile_array)

            output_path = output_paths[i]
            patch_image.save(output_path)
            logger.info(f"Saved test patch to {output_path}")
