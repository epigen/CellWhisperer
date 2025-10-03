#!/usr/bin/env python3
"""
Script to create a colored visualization of cells by cell type.
Uses the optimized coordinate transformation parameters from main.py.
"""

import openslide
from PIL import Image, ImageDraw
import os
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Configuration
LEVEL_INDEX = 1  # Level 1 corresponds to the second scale of the svs (0-based indexing)
svs_file = "/home/moritz/code/cellwhisperer/resources/lymphoma_cosmx_small/image.svs"
h5ad_file = "/home/moritz/code/cellwhisperer/resources/lymphoma_cosmx_small/read_count_table.h5ad"
output_raw_image = (
    "/home/moritz/code/cellwhisperer/resources/lymphoma_cosmx_small/level1_raw.png"
)
output_colored_mask = "/home/moritz/code/cellwhisperer/resources/lymphoma_cosmx_small/level1_colored_by_celltype.png"
output_overlayed_image = "/home/moritz/code/cellwhisperer/resources/lymphoma_cosmx_small/level1_with_colored_overlay.png"
output_legend = (
    "/home/moritz/code/cellwhisperer/resources/lymphoma_cosmx_small/celltype_legend.png"
)

# Optimized parameters from main.py
SCALING_FACTOR = 0.79488464508
X_OFFSET = 0.842708  # mm
Y_OFFSET = 1.235417  # mm
ROTATION_DEGREES = 269.73  # degrees

# Cell type color mapping - using distinct colors for different cell types
CELL_TYPE_COLORS = {
    "B cell": (0, 0, 255),  # Blue
    "T cell": (255, 0, 0),  # Red
    "NK cell": (0, 255, 0),  # Green
    "Macrophage": (255, 165, 0),  # Orange
    "Dendritic cell": (128, 0, 128),  # Purple
    "Neutrophil": (255, 255, 0),  # Yellow
    "Monocyte": (255, 192, 203),  # Pink
    "Plasma cell": (0, 255, 255),  # Cyan
    "Endothelial cell": (165, 42, 42),  # Brown
    "Fibroblast": (128, 128, 128),  # Gray
    "Epithelial cell": (255, 20, 147),  # Deep pink
    "Unknown": (64, 64, 64),  # Dark gray
}


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


def get_cell_type_from_adata(adata):
    """
    Extract cell type information from AnnData object.
    Tries common cell type annotation fields.

    Args:
        adata: AnnData object

    Returns:
        numpy array of cell type labels
    """
    # Common field names for cell types
    possible_fields = [
        "cell_type",
        "celltype",
        "cell_ontology_class",
        "leiden",
        "clusters",
        "annotation",
    ]

    for field in possible_fields:
        if field in adata.obs.columns:
            print(f"Found cell type information in field: {field}")
            return adata.obs[field].values

    # If no cell type field found, create dummy categories based on gene expression
    print(
        "No cell type field found. Creating dummy categories based on top expressed genes."
    )

    # Use top expressed genes to create rough categories
    if adata.X.shape[1] > 0:
        # Calculate mean expression per cell
        mean_expr = np.array(adata.X.mean(axis=1)).flatten()
        # Create 5 categories based on expression levels
        categories = np.digitize(mean_expr, np.percentile(mean_expr, [20, 40, 60, 80]))
        category_names = [
            "Low_expr",
            "Med_low_expr",
            "Medium_expr",
            "Med_high_expr",
            "High_expr",
        ]
        return np.array(
            [category_names[min(cat, len(category_names) - 1)] for cat in categories]
        )
    else:
        return np.array(["Unknown"] * adata.n_obs)


def assign_colors_to_cell_types(cell_types):
    """
    Assign colors to cell types, using predefined colors where available.

    Args:
        cell_types: Array of cell type labels

    Returns:
        Dictionary mapping cell types to RGB colors
    """
    unique_types = np.unique(cell_types)
    color_map = {}

    # Use predefined colors where available
    for cell_type in unique_types:
        if cell_type in CELL_TYPE_COLORS:
            color_map[cell_type] = CELL_TYPE_COLORS[cell_type]
        else:
            # Generate a random color for unknown cell types
            np.random.seed(
                hash(cell_type) % 2**32
            )  # Consistent color for same cell type
            color_map[cell_type] = tuple(np.random.randint(0, 256, 3))

    print(f"Assigned colors to {len(unique_types)} cell types:")
    for cell_type, color in color_map.items():
        print(f"  {cell_type}: RGB{color}")

    return color_map


def create_colored_cell_visualization(
    image, h5ad_path, image_dims, output_path, legend_path
):
    """
    Create a colored visualization where each cell is colored by its cell type.

    Args:
        image: PIL Image to overlay on
        h5ad_path: Path to h5ad file with cell coordinates and types
        image_dims: Tuple of (width, height) of the image
        output_path: Path for the output PNG file
        legend_path: Path for the legend PNG file

    Returns:
        PIL Image with colored cells
    """
    # Load the h5ad file
    adata = ad.read_h5ad(h5ad_path)

    # Extract coordinates
    x_coords_mm = adata.obs["x_slide_mm"].values
    y_coords_mm = adata.obs["y_slide_mm"].values

    # Get cell types
    cell_types = get_cell_type_from_adata(adata)

    print(
        f"Loaded {len(x_coords_mm)} cells with {len(np.unique(cell_types))} unique cell types"
    )

    # Assign colors to cell types
    color_map = assign_colors_to_cell_types(cell_types)

    # Apply coordinate transformation (same as optimized in main.py)
    x_coords_offset = x_coords_mm + X_OFFSET
    y_coords_offset = y_coords_mm + Y_OFFSET

    # Convert mm coordinates to pixel coordinates
    mm_to_pixel_x = (image_dims[0] / 10.0) * SCALING_FACTOR
    mm_to_pixel_y = (image_dims[1] / 10.0) * SCALING_FACTOR

    x_coords_px = (x_coords_offset * mm_to_pixel_x).astype(int)
    y_coords_px = (y_coords_offset * mm_to_pixel_y).astype(int)

    # Apply rotation around image center
    center_x = image_dims[0] / 2
    center_y = image_dims[1] / 2

    angle_rad = np.radians(ROTATION_DEGREES)
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
    cell_types_valid = cell_types[valid_mask]

    print(f"Found {len(x_coords_px)} valid cells within image bounds")
    print(
        f"Coordinate ranges: x=[{x_coords_px.min()}-{x_coords_px.max()}], y=[{y_coords_px.min()}-{y_coords_px.max()}]"
    )

    # Create colored visualization
    colored_image = Image.new(
        "RGBA", image.size, (0, 0, 0, 0)
    )  # Transparent background
    draw = ImageDraw.Draw(colored_image)

    # Draw circles for each cell, colored by cell type
    radius = 2
    for x, y, cell_type in zip(x_coords_px, y_coords_px, cell_types_valid):
        color = color_map[cell_type]
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(*color, 255),  # Full opacity
        )

    # Save the colored visualization (transparent background)
    colored_image.save(output_path, "PNG")
    print(f"Saved colored cell visualization to: {output_path}")

    # Create and save overlayed version with tissue background
    overlayed_path = output_path.replace(
        "_colored_by_celltype.png", "_with_colored_overlay.png"
    )
    create_overlayed_image(image, colored_image, overlayed_path)

    # Create and save legend
    create_legend(color_map, legend_path)

    return colored_image


def create_overlayed_image(tissue_image, colored_cells, output_path):
    """
    Create an overlayed image with colored cells on top of tissue image.

    Args:
        tissue_image: PIL Image of the tissue
        colored_cells: PIL Image with colored cells (RGBA with transparency)
        output_path: Path to save the overlayed image
    """
    # Convert tissue image to RGBA if needed
    if tissue_image.mode != "RGBA":
        tissue_copy = tissue_image.convert("RGBA")
    else:
        tissue_copy = tissue_image.copy()

    # Composite the colored cells onto the tissue image
    overlayed = Image.alpha_composite(tissue_copy, colored_cells)

    # Convert back to RGB for saving
    if overlayed.mode == "RGBA":
        rgb_image = Image.new("RGB", overlayed.size, (255, 255, 255))
        rgb_image.paste(overlayed, mask=overlayed.split()[-1])
        overlayed = rgb_image

    # Save the overlayed image
    overlayed.save(output_path, "PNG")
    print(f"Saved overlayed image to: {output_path}")


def create_legend(color_map, legend_path):
    """
    Create a legend showing cell types and their colors.

    Args:
        color_map: Dictionary mapping cell types to RGB colors
        legend_path: Path to save the legend image
    """
    # Create matplotlib figure for legend
    fig, ax = plt.subplots(figsize=(8, max(6, len(color_map) * 0.5)))

    # Create color patches for legend
    patches = []
    labels = []

    for cell_type, color in sorted(color_map.items()):
        # Normalize color to 0-1 range for matplotlib
        norm_color = tuple(c / 255.0 for c in color)
        patches.append(plt.Rectangle((0, 0), 1, 1, facecolor=norm_color))
        labels.append(cell_type)

    # Create legend
    ax.legend(patches, labels, loc="center", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Cell Type Color Legend", fontsize=16, fontweight="bold")

    # Save legend
    plt.tight_layout()
    plt.savefig(legend_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved color legend to: {legend_path}")


def print_parameters():
    """Print the coordinate transformation parameters being used."""
    print("\n" + "=" * 60)
    print("COORDINATE TRANSFORMATION PARAMETERS")
    print("=" * 60)
    print(f"Scaling factor: {SCALING_FACTOR}")
    print(f"X offset: {X_OFFSET} mm")
    print(f"Y offset: {Y_OFFSET} mm")
    print(f"Rotation: {ROTATION_DEGREES} degrees")
    print("=" * 60)
    print()


def main():
    """Main function to create colored cell type visualization."""
    if not os.path.exists(svs_file):
        print(f"Error: SVS file not found: {svs_file}")
        return 1

    if not os.path.exists(h5ad_file):
        print(f"Error: H5AD file not found: {h5ad_file}")
        return 1

    print(f"Processing SVS file: {svs_file}")
    print(f"Using cell data from: {h5ad_file}")

    # Print transformation parameters
    print_parameters()

    # Extract the raw level 1 image
    image, level_dims = extract_level_as_png(svs_file, output_raw_image, LEVEL_INDEX)

    # Create colored cell type visualization
    print(f"\nCreating colored cell type visualization...")
    colored_image = create_colored_cell_visualization(
        image, h5ad_file, level_dims, output_colored_mask, output_legend
    )

    print(f"\nCompleted successfully!")
    print(f"Raw image saved to: {output_raw_image}")
    print(f"Colored visualization saved to: {output_colored_mask}")
    print(f"Overlayed image saved to: {output_overlayed_image}")
    print(f"Color legend saved to: {output_legend}")

    return 0


if __name__ == "__main__":
    exit(main())
