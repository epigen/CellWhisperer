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
    filtered_coords, slide_wrapper, spot_diameter_pixels, output_paths, crop_tile_func, pixel_size_um=None, small_patch_diameter_um=None
):
    """Save random example patches for report with cell boundary overlay derived from parameters."""
    if len(output_paths) == 0:
        return

    # Calculate cell size from parameters if provided
    if pixel_size_um is not None and small_patch_diameter_um is not None:
        cell_size = int(round(small_patch_diameter_um / pixel_size_um))
        logger.info(f"Saving {len(output_paths)} random test patches with {cell_size}px cell boundaries (derived from {small_patch_diameter_um}μm / {pixel_size_um}μm/px).")
    else:
        cell_size = 56  # Fallback to default
        logger.info(f"Saving {len(output_paths)} random test patches with {cell_size}px cell boundaries (default).")
    
    # Ensure we don't try to sample more tiles than available
    num_patches_to_save = min(len(filtered_coords), len(output_paths))

    if num_patches_to_save > 0:
        # Randomly sample tiles
        sample_tiles = filtered_coords.sample(n=num_patches_to_save, random_state=42)

        for i, (index, row) in enumerate(sample_tiles.iterrows()):
            x = int(row.x_pixel)
            y = int(row.y_pixel)

            tile_array = crop_tile_func(slide_wrapper, x, y, spot_diameter_pixels)
            # Ensure RGB (drop alpha if present)
            tile_array = tile_array[:, :, :3]
            
            # Create matplotlib figure to add cell boundary overlay
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(tile_array)
            
            # Draw cell boundary in the center of the patch
            patch_center = spot_diameter_pixels // 2
            cell_half = cell_size // 2
            
            # Calculate cell boundary rectangle (centered in the patch)
            cell_rect = patches.Rectangle(
                (patch_center - cell_half, patch_center - cell_half),
                cell_size,
                cell_size,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                alpha=0.8,
            )
            ax.add_patch(cell_rect)
            
            # Remove axes and save
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, spot_diameter_pixels)
            ax.set_ylim(spot_diameter_pixels, 0)  # Flip y-axis to match image coordinates
            
            output_path = output_paths[i]
            plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            logger.info(f"Saved test patch with {cell_size}px cell boundary to {output_path}")
