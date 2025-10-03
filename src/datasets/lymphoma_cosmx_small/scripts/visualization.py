"""Visualization utilities for lymphoma CosMx data processing."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def generate_qc_tile_plot(
    coord_df, img_np, spot_diameter_pixels, scale_factor_img, output_path
):
    """Generate QC plot showing tile selection."""
    logger.info("Generating QC tile plot...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_np)
    scaled_spot_diameter = spot_diameter_pixels * scale_factor_img

    for _, row in coord_df.iterrows():
        x = row.x_pixel * scale_factor_img
        y = row.y_pixel * scale_factor_img
        color = "red" if row.is_white else "blue"
        rect = patches.Rectangle(
            (x, y),
            scaled_spot_diameter,
            scaled_spot_diameter,
            linewidth=0.5,
            edgecolor=color,
            facecolor="none",
            alpha=0.5,
        )
        ax.add_patch(rect)
    ax.set_title("Tile Selection (Blue=Kept, Red=Discarded)")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved QC tile plot to {output_path}")


def save_example_patches(
    filtered_coords, slide_wrapper, spot_diameter_pixels, output_paths, crop_tile_func
):
    """Save random example patches for report."""
    if len(output_paths) == 0:
        return

    logger.info(f"Saving {len(output_paths)} random test patches.")
    # Ensure we don't try to sample more tiles than available
    num_patches_to_save = min(len(filtered_coords), len(output_paths))

    if num_patches_to_save > 0:
        # Randomly sample tiles
        sample_tiles = filtered_coords.sample(n=num_patches_to_save, random_state=42)

        for i, (index, row) in enumerate(sample_tiles.iterrows()):
            x = int(row.x_pixel)
            y = int(row.y_pixel)

            tile_array = crop_tile_func(slide_wrapper, x, y, spot_diameter_pixels)
            patch_image = Image.fromarray(tile_array)

            output_path = output_paths[i]
            patch_image.save(output_path)
            logger.info(f"Saved test patch to {output_path}")
