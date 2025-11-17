"""Coordinate transformation utilities for lymphoma CosMx data processing."""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def convert_mm_to_pixel_coordinates(
    adata, image_width, image_height, x_offset, y_offset, scale_factor, rotation_degrees=269.73
):
    """Convert mm coordinates to pixel coordinates using rotation-based transformation.

    This implements the optimized coordinate transformation from main.py:
    1. Apply offsets to mm coordinates
    2. Convert to pixel coordinates using scale factor
    3. Apply rotation around image center

    Args:
        adata: AnnData object with x_slide_mm and y_slide_mm fields
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        x_offset: X offset to apply (0.842708)
        y_offset: Y offset to apply (1.235417)
        scale_factor: Scale factor (0.79488464508)
        rotation_degrees: Rotation angle in degrees (269.73)

    Returns:
        numpy array of shape (n_cells, 2) with [x_pixel, y_pixel] coordinates
    """
    # Check if we have the required mm coordinates
    if "x_slide_mm" not in adata.obs or "y_slide_mm" not in adata.obs:
        raise ValueError(
            "AnnData object must have 'x_slide_mm' and 'y_slide_mm' fields in .obs"
        )

    x_mm = adata.obs["x_slide_mm"].values
    y_mm = adata.obs["y_slide_mm"].values

    logger.info(f"Converting {len(x_mm)} cells from mm to pixel coordinates")
    logger.info(f"x_mm range: [{x_mm.min():.3f}, {x_mm.max():.3f}]")
    logger.info(f"y_mm range: [{y_mm.min():.3f}, {y_mm.max():.3f}]")

    # Step 1: Apply offsets
    x_coords_offset = x_mm + x_offset
    y_coords_offset = y_mm + y_offset

    # Step 2: Convert to pixel coordinates
    mm_to_pixel_x = (image_width / 10.0) * scale_factor
    mm_to_pixel_y = (image_height / 10.0) * scale_factor

    logger.info(f"mm_to_pixel_x: {mm_to_pixel_x:.3f}")
    logger.info(f"mm_to_pixel_y: {mm_to_pixel_y:.3f}")

    x_coords_px = x_coords_offset * mm_to_pixel_x
    y_coords_px = y_coords_offset * mm_to_pixel_y

    # Step 3: Apply rotation around image center
    center_x = image_width / 2
    center_y = image_height / 2

    # Convert rotation angle to radians
    angle_rad = np.radians(rotation_degrees)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    logger.info(f"Applying {rotation_degrees:.2f}° rotation around center ({center_x:.1f}, {center_y:.1f})")

    # Translate to origin, rotate, then translate back
    x_centered = x_coords_px - center_x
    y_centered = y_coords_px - center_y

    x_rotated = x_centered * cos_angle - y_centered * sin_angle
    y_rotated = x_centered * sin_angle + y_centered * cos_angle

    x_coords_final = x_rotated + center_x
    y_coords_final = y_rotated + center_y

    # Combine into coordinate array
    spatial_coords = np.column_stack([x_coords_final, y_coords_final])

    logger.info(
        f"Converted pixel coordinates range: x=[{x_coords_final.min():.1f}, {x_coords_final.max():.1f}], y=[{y_coords_final.min():.1f}, {y_coords_final.max():.1f}]"
    )

    return spatial_coords
