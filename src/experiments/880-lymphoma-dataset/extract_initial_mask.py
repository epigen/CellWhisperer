#!/usr/bin/env python3
"""
Script to extract level 1 from SVS file and create initial mask overlay.
Saves both the raw image and the image with cell mask overlay.
"""

import openslide
from PIL import Image, ImageDraw
import os
import anndata as ad
import numpy as np

# Configuration
LEVEL_INDEX = 1  # Level 1 corresponds to the second scale of the svs (0-based indexing)
svs_file = "/home/moritz/code/cellwhisperer/resources/lymphoma_cosmx_small/image.svs"
h5ad_file = "/home/moritz/code/cellwhisperer/resources/lymphoma_cosmx_small/read_count_table.h5ad"
output_raw_image = (
    "/home/moritz/code/cellwhisperer/resources/lymphoma_cosmx_small/level1_raw.png"
)
output_mask_image = "/home/moritz/code/cellwhisperer/resources/lymphoma_cosmx_small/level1_mask_only.png"

# Initial parameters for mask overlay (optimized from main.py)
INITIAL_SCALING_FACTOR = 0.79488464508
INITIAL_X_OFFSET = 0.842708  # mm
INITIAL_Y_OFFSET = 1.235417  # mm
INITIAL_ROTATION = 269.73  # degrees


def extract_level_as_png(svs_path, output_path, level_index=1):
    """
    Extract a specific level from an SVS file and save as PNG.

    Args:
        svs_path: Path to the input SVS file
        output_path: Path for the output PNG file
        level_index: Which level to extract (default: 1 for second scale)

    Returns:
        Tuple of (PIL Image, level_dimensions)
    """
    # Open the SVS file
    slide = openslide.OpenSlide(svs_path)

    # Get level information
    level_count = slide.level_count
    level_dimensions = slide.level_dimensions
    level_downsamples = slide.level_downsamples

    print(f"Available levels: {level_count}")
    print(f"Level dimensions: {level_dimensions}")
    print(f"Level downsamples: {level_downsamples}")

    # Check if requested level exists
    if level_index >= level_count:
        raise ValueError(
            f"SVS file only has {level_count} levels, but level {level_index} was requested"
        )

    # Extract the requested level
    level_dims = level_dimensions[level_index]
    print(f"Extracting level {level_index} with dimensions: {level_dims}")

    # Read the entire level image
    image = slide.read_region((0, 0), level_index, level_dims)

    # Convert RGBA to RGB (SVS files often have alpha channel)
    if image.mode == "RGBA":
        # Create white background
        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
        image = rgb_image

    # Save as PNG
    image.save(output_path, "PNG")
    print(f"Saved level {level_index} raw image to: {output_path}")

    # Close the slide
    slide.close()

    return image, level_dims


def create_initial_mask_overlay(image, h5ad_path, image_dims, output_path):
    """
    Create a simple mask overlay using initial parameters (no optimization).

    Args:
        image: PIL Image to overlay mask on
        h5ad_path: Path to h5ad file with cell coordinates
        image_dims: Tuple of (width, height) of the image
        output_path: Path for the output PNG file with mask

    Returns:
        PIL Image with mask overlay
    """
    # Load the h5ad file
    adata = ad.read_h5ad(h5ad_path)

    # Extract coordinates
    x_coords_mm = adata.obs["x_slide_mm"].values
    y_coords_mm = adata.obs["y_slide_mm"].values

    print(f"Loaded {len(x_coords_mm)} cell coordinates from h5ad file")

    # Apply initial offsets to coordinates
    x_coords_offset = x_coords_mm + INITIAL_X_OFFSET
    y_coords_offset = y_coords_mm + INITIAL_Y_OFFSET

    # Convert mm coordinates to pixel coordinates with initial factor
    mm_to_pixel_x = (image_dims[0] / 10.0) * INITIAL_SCALING_FACTOR
    mm_to_pixel_y = (image_dims[1] / 10.0) * INITIAL_SCALING_FACTOR

    # Convert mm coordinates to pixel coordinates
    x_coords_px = (x_coords_offset * mm_to_pixel_x).astype(int)
    y_coords_px = (y_coords_offset * mm_to_pixel_y).astype(int)

    # Apply rotation around image center
    center_x = image_dims[0] / 2
    center_y = image_dims[1] / 2

    # Convert rotation angle to radians
    angle_rad = np.radians(INITIAL_ROTATION)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Translate to origin, rotate, then translate back
    x_centered = x_coords_px - center_x
    y_centered = y_coords_px - center_y

    x_rotated = x_centered * cos_angle - y_centered * sin_angle
    y_rotated = x_centered * sin_angle + y_centered * cos_angle

    x_coords_px = (x_rotated + center_x).astype(int)
    y_coords_px = (y_rotated + center_y).astype(int)

    # Filter coordinates to be within image bounds
    valid_mask = (
        (x_coords_px >= 0)
        & (x_coords_px < image_dims[0])
        & (y_coords_px >= 0)
        & (y_coords_px < image_dims[1])
    )

    x_coords_px = x_coords_px[valid_mask]
    y_coords_px = y_coords_px[valid_mask]

    print(f"Found {len(x_coords_px)} valid cells within image bounds")
    print(
        f"Coordinate ranges: x=[{x_coords_px.min()}-{x_coords_px.max()}], y=[{y_coords_px.min()}-{y_coords_px.max()}]"
    )

    # Create a transparent background for the mask
    mask_image = Image.new("RGBA", image.size, (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(mask_image)

    # Draw circles for each cell position
    radius = 2  # Small radius for cell markers
    for x, y in zip(x_coords_px, y_coords_px):
        # Draw bright green circle on transparent background
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(0, 255, 0, 255),  # Bright green, fully opaque
        )

    # Save the mask image
    mask_image.save(output_path, "PNG")
    print(f"Saved mask only to: {output_path}")

    return mask_image


def print_initial_parameters():
    """Print the initial parameters being used for the mask overlay."""
    print("\n" + "=" * 60)
    print("INITIAL MASK PARAMETERS")
    print("=" * 60)
    print(f"Scaling factor: {INITIAL_SCALING_FACTOR}")
    print(f"X offset: {INITIAL_X_OFFSET} mm")
    print(f"Y offset: {INITIAL_Y_OFFSET} mm")
    print(f"Rotation: {INITIAL_ROTATION} degrees")
    print("=" * 60)
    print()


def main():
    """Main function to extract SVS level and create initial mask."""
    if not os.path.exists(svs_file):
        print(f"Error: SVS file not found: {svs_file}")
        return 1

    if not os.path.exists(h5ad_file):
        print(f"Error: H5AD file not found: {h5ad_file}")
        return 1

    print(f"Processing SVS file: {svs_file}")
    print(f"Using cell data from: {h5ad_file}")

    # Print initial parameters
    print_initial_parameters()

    # Extract the raw level 1 image
    image, level_dims = extract_level_as_png(svs_file, output_raw_image, LEVEL_INDEX)

    # Create mask overlay with initial parameters
    print(f"\nCreating initial mask overlay...")
    masked_image = create_initial_mask_overlay(
        image, h5ad_file, level_dims, output_mask_image
    )

    print(f"\nCompleted successfully!")
    print(f"Raw image saved to: {output_raw_image}")
    print(f"Mask only saved to: {output_mask_image}")

    return 0


if __name__ == "__main__":
    exit(main())
