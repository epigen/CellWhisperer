# TODO runthree iterations. first rough than coarser then ultra coarse

import openslide
from PIL import Image, ImageDraw
import os
import anndata as ad
import numpy as np
import subprocess
import tempfile
import time

LEVEL_INDEX = 1  # Level 1 corresponds to the second scale of the svs (0-based indexing)
svs_file = "/home/moritz/code/cellwhisperer/resources/lymphoma_cosmx_small/image.svs"
h5ad_file = "/home/moritz/code/cellwhisperer/resources/lymphoma_cosmx_small/read_count_table.h5ad"
output_file = "/home/moritz/code/cellwhisperer/resources/lymphoma_cosmx_small/CosMx_1K_CART_1__61360_level2_with_mask.png"

FIXED_SCALING_FACTOR = 0.79488464508  # determined from previous runs
FIXED_ROTATION_DEGREES = (
    270 - 0.27
)  # 269.73 degrees  # not sure if correct direction. maybe + 0.27 is better
X_OFFSET_RANGE = (0.8, 0.9)
Y_OFFSET_RANGE = (
    1.2,
    1.4,
)  # be careful, due to 270 degree rotation, y goes to -x direction
NUM_ITERATIONS = 4


def extract_second_scale(svs_path, output_path, h5ad_path=None):
    """
    Extract the second scale (level 2) from an SVS file and save as PNG.
    Optionally overlay cell positions from h5ad file as green mask.

    Args:
        svs_path: Path to the input SVS file
        output_path: Path for the output PNG file
        h5ad_path: Optional path to h5ad file with cell coordinates
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

    # Check if level 2 (second scale) exists
    if level_count < 3:
        raise ValueError(
            f"SVS file only has {level_count} levels, but level 2 (second scale) was requested"
        )

    # Extract level 1 (index 1, which is the second scale)
    level_dims = level_dimensions[LEVEL_INDEX]
    # Store level 0 dimensions for final equation computation
    level_0_dims = level_dimensions[0]

    print(f"Extracting level {LEVEL_INDEX} with dimensions: {level_dims}")
    print(f"Level 0 (full resolution) dimensions: {level_0_dims}")

    # Read the entire level 1 image
    image = slide.read_region((0, 0), LEVEL_INDEX, level_dims)

    # Convert RGBA to RGB (SVS files often have alpha channel)
    if image.mode == "RGBA":
        # Create white background
        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
        image = rgb_image

    # Add cell mask overlay if h5ad file is provided
    if h5ad_path and os.path.exists(h5ad_path):
        print(f"Loading cell data from: {h5ad_path}")
        image = add_cell_mask(image, h5ad_path, level_dims, level_0_dims, output_path)

    # Save as PNG
    image.save(output_path, "PNG")
    print(f"Saved level {LEVEL_INDEX} image to: {output_path}")

    # Close the slide
    slide.close()


def find_optimal_parameters_iterative(
    image,
    h5ad_path,
    image_dims,
    x_offset_range,
    y_offset_range,
    iteration_name="",
):
    """
    Find the optimal x and y offsets by testing different combinations and measuring pixel intensity.
    Uses fixed scaling factor and rotation angle.

    Args:
        image: PIL Image to test on
        h5ad_path: Path to h5ad file with cell coordinates
        image_dims: Tuple of (width, height) of the image
        x_offset_range: Tuple of (min, max) for x offset in mm
        y_offset_range: Tuple of (min, max) for y offset in mm
        iteration_name: String to identify the current iteration

    Returns:
        Tuple of (optimal_x_offset_mm, optimal_y_offset_mm, min_intensity)
    """
    # Load the h5ad file
    adata = ad.read_h5ad(h5ad_path)

    # Extract coordinates
    x_coords_mm = adata.obs["x_slide_mm"].values
    y_coords_mm = adata.obs["y_slide_mm"].values

    # Convert image to grayscale for intensity calculation
    gray_image = image.convert("L")
    image_array = np.array(gray_image)

    # Test parameters: 4 steps each for x-offset, y-offset (16 total combinations)
    x_offsets_mm = np.linspace(*x_offset_range, NUM_ITERATIONS)
    y_offsets_mm = np.linspace(*y_offset_range, NUM_ITERATIONS)

    min_intensity = float("inf")
    best_x_offset = x_offset_range[0]
    best_y_offset = y_offset_range[0]

    total_combinations = len(x_offsets_mm) * len(y_offsets_mm)
    print(f"\n{iteration_name}: Testing {total_combinations} parameter combinations...")
    print(f"Fixed scaling factor: {FIXED_SCALING_FACTOR}")
    print(f"Fixed rotation: {FIXED_ROTATION_DEGREES} degrees")
    print(f"X offset range: {x_offset_range}")
    print(f"Y offset range: {y_offset_range}")

    combination_count = 0
    for j, x_offset_mm in enumerate(x_offsets_mm):
        for k, y_offset_mm in enumerate(y_offsets_mm):
            combination_count += 1

            # Apply offsets to coordinates
            x_coords_offset = x_coords_mm + x_offset_mm
            y_coords_offset = y_coords_mm + y_offset_mm

            # Convert mm coordinates to pixel coordinates with fixed factor
            mm_to_pixel_x = (image_dims[0] / 10.0) * FIXED_SCALING_FACTOR
            mm_to_pixel_y = (image_dims[1] / 10.0) * FIXED_SCALING_FACTOR

            # Convert mm coordinates to pixel coordinates
            x_coords_px = (x_coords_offset * mm_to_pixel_x).astype(int)
            y_coords_px = (y_coords_offset * mm_to_pixel_y).astype(int)

            # Apply rotation around image center
            center_x = image_dims[0] / 2
            center_y = image_dims[1] / 2

            # Convert rotation angle to radians
            angle_rad = np.radians(FIXED_ROTATION_DEGREES)
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

            x_coords_valid = x_coords_px[valid_mask]
            y_coords_valid = y_coords_px[valid_mask]

            # Calculate sum of pixel intensities at cell locations
            total_intensity = np.int64(0)
            if len(x_coords_valid) > 0:
                sample_radius = 1
                for x, y in zip(x_coords_valid, y_coords_valid):
                    y_min = max(0, y - sample_radius)
                    y_max = min(image_array.shape[0], y + sample_radius + 1)
                    x_min = max(0, x - sample_radius)
                    x_max = min(image_array.shape[1], x + sample_radius + 1)

                    region = image_array[y_min:y_max, x_min:x_max]
                    if region.size > 0:
                        region_intensity = np.mean(region) * region.size
                        total_intensity += np.int64(region_intensity)

            # Update best parameters if this combination is better (lower intensity)
            if total_intensity < min_intensity:
                min_intensity = total_intensity
                best_x_offset = x_offset_mm
                best_y_offset = y_offset_mm

            if combination_count % 4 == 0:  # Progress update every 4 combinations
                print(f"  Progress: {combination_count}/{total_combinations}")

    print(f"{iteration_name} - Fixed scaling factor: {FIXED_SCALING_FACTOR}")
    print(f"{iteration_name} - Optimal x offset: {best_x_offset:.6f} mm")
    print(f"{iteration_name} - Optimal y offset: {best_y_offset:.6f} mm")
    print(f"{iteration_name} - Fixed rotation: {FIXED_ROTATION_DEGREES} degrees")
    print(f"{iteration_name} - Minimum intensity sum: {min_intensity}")

    return best_x_offset, best_y_offset, min_intensity


def show_image_with_overlay(
    image,
    x_coords_px,
    y_coords_px,
    factor,
    x_offset,
    y_offset,
    intensity,
    combo_num,
    total_combos,
):
    """
    Show an image with cell mask overlay by saving to temp file and opening with external viewer.

    Args:
        image: PIL Image to display
        x_coords_px: Array of x coordinates in pixels
        y_coords_px: Array of y coordinates in pixels
        factor: Scaling factor used
        x_offset: X offset in mm
        y_offset: Y offset in mm
        intensity: Total intensity sum
        combo_num: Current combination number
        total_combos: Total number of combinations
    """
    # Create a copy of the image to overlay on
    display_image = image.copy()

    # Create a transparent overlay
    overlay = Image.new("RGBA", display_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw circles for each cell position
    radius = 2  # Slightly larger radius for better visibility
    for x, y in zip(x_coords_px, y_coords_px):
        # Draw semi-transparent green circle
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(0, 255, 0, 128),  # Semi-transparent green
        )

    # Convert base image to RGBA if needed
    if display_image.mode != "RGBA":
        display_image = display_image.convert("RGBA")

    # Composite the overlay onto the image
    result_image = Image.alpha_composite(display_image, overlay)

    # Convert back to RGB for saving
    if result_image.mode == "RGBA":
        rgb_image = Image.new("RGB", result_image.size, (255, 255, 255))
        rgb_image.paste(result_image, mask=result_image.split()[-1])
        result_image = rgb_image

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        temp_path = tmp_file.name

    # Save the image
    result_image.save(temp_path, "PNG")

    # Print information about current combination
    print(
        f"Combo {combo_num}/{total_combos}: Factor={factor:.4f}, X_offset={x_offset:.4f}mm, Y_offset={y_offset:.4f}mm"
    )
    print(f"Cells: {len(x_coords_px)}, Intensity: {intensity}")
    print(f"Saved temp image: {temp_path}")

    # Open with feh and wait for it to close
    try:
        subprocess.run(["feh", "--scale-down", temp_path], check=True)
    except subprocess.CalledProcessError:
        print(f"Could not open image with feh. Image saved at: {temp_path}")
        input("Press Enter to continue to next combination...")
    except FileNotFoundError:
        print("feh not found. Please install feh or view the image manually.")
        print(f"Image saved at: {temp_path}")
        input("Press Enter to continue to next combination...")
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def iterative_parameter_refinement(image, h5ad_path, image_dims, output_path):
    """
    Perform iterative refinement of x/y offset parameters over 3 rounds.
    Uses fixed scaling factor and rotation angle.

    Args:
        image: PIL Image to test on
        h5ad_path: Path to h5ad file with cell coordinates
        image_dims: Tuple of (width, height) of the image
        output_path: Base path for output files

    Returns:
        Tuple of final optimal parameters (x_offset, y_offset)
    """
    # Initial ranges
    x_offset_range = X_OFFSET_RANGE
    y_offset_range = Y_OFFSET_RANGE

    best_x_offset = None
    best_y_offset = None

    for iteration in range(3):
        iteration_name = f"Iteration {iteration + 1}/3"

        # Find optimal parameters for current ranges
        best_x_offset, best_y_offset, min_intensity = find_optimal_parameters_iterative(
            image,
            h5ad_path,
            image_dims,
            x_offset_range,
            y_offset_range,
            iteration_name,
        )

        # Save and show image with current best parameters
        iteration_output_path = output_path.replace(
            ".png",
            f"_iteration_{iteration + 1}_factor_{FIXED_SCALING_FACTOR:.5f}_x{best_x_offset:.6f}_y{best_y_offset:.6f}_rot{FIXED_ROTATION_DEGREES:.2f}.png",
        )

        save_image_with_optimal_parameters(
            image,
            h5ad_path,
            image_dims,
            FIXED_SCALING_FACTOR,
            best_x_offset,
            best_y_offset,
            FIXED_ROTATION_DEGREES,
            iteration_output_path,
        )

        print(f"\n{iteration_name} complete. Displaying result...")
        try:
            subprocess.Popen(["feh", "--scale-down", iteration_output_path])
        except FileNotFoundError:
            print(f"feh not found. Image saved at: {iteration_output_path}")

        # Prepare ranges for next iteration (narrow down around best values)
        if iteration < 2:  # Don't narrow for the last iteration
            # Calculate range widths for next iteration (reduce by factor of 4)
            x_offset_width = (x_offset_range[1] - x_offset_range[0]) / 4
            y_offset_width = (y_offset_range[1] - y_offset_range[0]) / 4

            # Center new ranges around best values
            x_offset_range = (
                best_x_offset - x_offset_width / 2,
                best_x_offset + x_offset_width / 2,
            )
            y_offset_range = (
                best_y_offset - y_offset_width / 2,
                best_y_offset + y_offset_width / 2,
            )

            print(f"\nNarrowing ranges for next iteration:")
            print(
                f"  New x offset range: ({x_offset_range[0]:.6f}, {x_offset_range[1]:.6f})"
            )
            print(
                f"  New y offset range: ({y_offset_range[0]:.6f}, {y_offset_range[1]:.6f})"
            )

    return best_x_offset, best_y_offset


def add_cell_mask(image, h5ad_path, image_dims, level_0_dims, output_path):
    """
    Add semi-transparent green mask based on cell positions from h5ad file.
    Uses iterative refinement to find optimal alignment parameters.

    Args:
        image: PIL Image to overlay mask on
        h5ad_path: Path to h5ad file with cell coordinates
        image_dims: Tuple of (width, height) of the working image (level 1)
        level_0_dims: Tuple of (width, height) of the full resolution image (level 0)
        output_path: Path for the output PNG file

    Returns:
        PIL Image with mask overlay
    """
    # Load the h5ad file
    adata = ad.read_h5ad(h5ad_path)

    # Extract coordinates
    x_coords_mm = adata.obs["x_slide_mm"].values
    y_coords_mm = adata.obs["y_slide_mm"].values

    # Show a preview with initial parameters to verify rotation is roughly correct
    print("Creating preview image with initial parameters to verify rotation...")
    preview_output_path = output_path.replace(".png", "_PREVIEW_rotation_check.png")

    # Use middle values of the ranges for preview
    preview_x_offset = (X_OFFSET_RANGE[0] + X_OFFSET_RANGE[1]) / 2
    preview_y_offset = (Y_OFFSET_RANGE[0] + Y_OFFSET_RANGE[1]) / 2

    save_image_with_optimal_parameters(
        image,
        h5ad_path,
        image_dims,
        FIXED_SCALING_FACTOR,
        preview_x_offset,
        preview_y_offset,
        FIXED_ROTATION_DEGREES,
        preview_output_path,
    )

    print(f"Preview saved to: {preview_output_path}")
    print(
        f"Preview parameters: factor={FIXED_SCALING_FACTOR:.5f}, x_offset={preview_x_offset:.3f}, y_offset={preview_y_offset:.3f}, rotation={FIXED_ROTATION_DEGREES:.2f}"
    )
    print("Please check if the rotation looks roughly correct before proceeding...")

    try:
        subprocess.Popen(["feh", "--scale-down", preview_output_path])
    except FileNotFoundError:
        print(f"feh not found. Image saved at: {preview_output_path}")

    response = (
        input("Does the line-up look very roughly correct? (y/n): ").strip().lower()
    )
    if len(response) > 0 and response[0] == "n":
        print("Please adjust the ranges at the top of the script and try again.")
        return image

    # Perform iterative parameter refinement
    optimal_x_offset, optimal_y_offset = iterative_parameter_refinement(
        image, h5ad_path, image_dims, output_path
    )

    # Print the mapping equation using level 0 dimensions for final computation
    print_mapping_equation(
        FIXED_SCALING_FACTOR,
        optimal_x_offset,
        optimal_y_offset,
        FIXED_ROTATION_DEGREES,
        level_0_dims,  # Use full resolution dimensions for final equation
    )

    # Save final image with optimal parameters
    final_output_path = output_path.replace(
        ".png",
        f"_FINAL_factor_{FIXED_SCALING_FACTOR:.5f}_x{optimal_x_offset:.6f}_y{optimal_y_offset:.6f}_rot{FIXED_ROTATION_DEGREES:.2f}.png",
    )
    save_image_with_optimal_parameters(
        image,
        h5ad_path,
        image_dims,
        FIXED_SCALING_FACTOR,
        optimal_x_offset,
        optimal_y_offset,
        FIXED_ROTATION_DEGREES,
        final_output_path,
    )

    # Apply optimal parameters to create the returned image
    x_coords_offset = x_coords_mm + optimal_x_offset
    y_coords_offset = y_coords_mm + optimal_y_offset

    mm_to_pixel_x = (image_dims[0] / 10.0) * FIXED_SCALING_FACTOR
    mm_to_pixel_y = (image_dims[1] / 10.0) * FIXED_SCALING_FACTOR

    # Convert mm coordinates to pixel coordinates
    x_coords_px = (x_coords_offset * mm_to_pixel_x).astype(int)
    y_coords_px = (y_coords_offset * mm_to_pixel_y).astype(int)

    # Apply rotation around image center
    center_x = image_dims[0] / 2
    center_y = image_dims[1] / 2

    # Convert rotation angle to radians
    angle_rad = np.radians(FIXED_ROTATION_DEGREES)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Translate to origin, rotate, then translate back
    x_centered = x_coords_px - center_x
    y_centered = y_coords_px - center_y

    x_rotated = x_centered * cos_angle - y_centered * sin_angle
    y_rotated = x_centered * sin_angle + y_centered * cos_angle

    x_coords_px = (x_rotated + center_x).astype(int)
    y_coords_px = (y_rotated + center_y).astype(int)

    valid_mask = (
        (x_coords_px >= 0)
        & (x_coords_px < image_dims[0])
        & (y_coords_px >= 0)
        & (y_coords_px < image_dims[1])
    )

    x_coords_px = x_coords_px[valid_mask]
    y_coords_px = y_coords_px[valid_mask]

    print(f"Final: Found {len(x_coords_px)} valid cells within image bounds")

    # Create overlay with optimal parameters
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    radius = 2
    for x, y in zip(x_coords_px, y_coords_px):
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(0, 255, 0, 128),
        )

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    image = Image.alpha_composite(image, overlay)

    if image.mode == "RGBA":
        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[-1])
        image = rgb_image

    return image


def print_mapping_equation(
    optimal_factor, optimal_x_offset, optimal_y_offset, optimal_rotation, image_dims
):
    """
    Print the mathematical equation for mapping mm coordinates to pixel coordinates
    using the optimal parameters.

    Args:
        optimal_factor: The optimal scaling factor
        optimal_x_offset: The optimal x offset in mm
        optimal_y_offset: The optimal y offset in mm
        optimal_rotation: The optimal rotation angle in degrees
        image_dims: Tuple of (width, height) of the full resolution image (level 0)
    """
    # Calculate the conversion factors
    mm_to_pixel_x = (image_dims[0] / 10.0) * optimal_factor
    mm_to_pixel_y = (image_dims[1] / 10.0) * optimal_factor

    # Calculate center coordinates
    center_x = image_dims[0] / 2
    center_y = image_dims[1] / 2

    print("\n" + "=" * 80)
    print("COORDINATE MAPPING EQUATION (FOR FULL RESOLUTION LEVEL 0 IMAGE)")
    print("=" * 80)
    print("Given input coordinates (x_mm, y_mm) in millimeters:")
    print()
    print("Step 1: Apply offsets")
    print(f"  x_offset = x_mm + {optimal_x_offset:.6f}")
    print(f"  y_offset = y_mm + {optimal_y_offset:.6f}")
    print()
    print("Step 2: Convert to pixel coordinates")
    print(
        f"  mm_to_pixel_x = ({image_dims[0]} / 10.0) * {optimal_factor:.6f} = {mm_to_pixel_x:.6f}"
    )
    print(
        f"  mm_to_pixel_y = ({image_dims[1]} / 10.0) * {optimal_factor:.6f} = {mm_to_pixel_y:.6f}"
    )
    print()
    print("  x_px_temp = x_offset * mm_to_pixel_x")
    print("  y_px_temp = y_offset * mm_to_pixel_y")
    print()
    print(f"Step 3: Apply {optimal_rotation:.6f}-degree rotation around image center")
    print(f"  center_x = {center_x:.1f}")
    print(f"  center_y = {center_y:.1f}")
    print(
        f"  angle = {optimal_rotation:.6f} degrees = {np.radians(optimal_rotation):.6f} radians"
    )
    print("  cos_angle = {:.6f}".format(np.cos(np.radians(optimal_rotation))))
    print("  sin_angle = {:.6f}".format(np.sin(np.radians(optimal_rotation))))
    print()
    print("  x_centered = x_px_temp - center_x")
    print("  y_centered = y_px_temp - center_y")
    print("  x_rotated = x_centered * cos_angle - y_centered * sin_angle")
    print("  y_rotated = x_centered * sin_angle + y_centered * cos_angle")
    print("  x_px_final = x_rotated + center_x")
    print("  y_px_final = y_rotated + center_y")
    print()
    print("COMPLETE EQUATION:")
    print("----------------")
    cos_angle = np.cos(np.radians(optimal_rotation))
    sin_angle = np.sin(np.radians(optimal_rotation))
    print("For input coordinates (x_mm, y_mm):")
    print(f"x_px_temp = (x_mm + {optimal_x_offset:.6f}) * {mm_to_pixel_x:.6f}")
    print(f"y_px_temp = (y_mm + {optimal_y_offset:.6f}) * {mm_to_pixel_y:.6f}")
    print(f"x_centered = x_px_temp - {center_x:.1f}")
    print(f"y_centered = y_px_temp - {center_y:.1f}")
    print(
        f"x_px_final = x_centered * {cos_angle:.6f} - y_centered * {sin_angle:.6f} + {center_x:.1f}"
    )
    print(
        f"y_px_final = x_centered * {sin_angle:.6f} + y_centered * {cos_angle:.6f} + {center_y:.1f}"
    )
    print("=" * 80)
    print()


def save_image_with_optimal_parameters(
    image,
    h5ad_path,
    image_dims,
    optimal_factor,
    optimal_x_offset,
    optimal_y_offset,
    optimal_rotation,
    output_path,
):
    """
    Save an image with cell mask using the optimal scaling factor, offsets, and rotation.

    Args:
        image: PIL Image to overlay mask on
        h5ad_path: Path to h5ad file with cell coordinates
        image_dims: Tuple of (width, height) of the image
        optimal_factor: The optimal scaling factor to use
        optimal_x_offset: The optimal x offset in mm
        optimal_y_offset: The optimal y offset in mm
        optimal_rotation: The optimal rotation angle in degrees
        output_path: Path to save the image with optimal parameters
    """
    # Load the h5ad file
    adata = ad.read_h5ad(h5ad_path)

    # Extract coordinates
    x_coords_mm = adata.obs["x_slide_mm"].values
    y_coords_mm = adata.obs["y_slide_mm"].values

    # Apply offsets to coordinates
    x_coords_offset = x_coords_mm + optimal_x_offset
    y_coords_offset = y_coords_mm + optimal_y_offset

    # Convert mm coordinates to pixel coordinates with optimal factor
    mm_to_pixel_x = (image_dims[0] / 10.0) * optimal_factor
    mm_to_pixel_y = (image_dims[1] / 10.0) * optimal_factor

    # Convert mm coordinates to pixel coordinates
    x_coords_px = (x_coords_offset * mm_to_pixel_x).astype(int)
    y_coords_px = (y_coords_offset * mm_to_pixel_y).astype(int)

    # Apply rotation around image center
    center_x = image_dims[0] / 2
    center_y = image_dims[1] / 2

    # Convert rotation angle to radians
    angle_rad = np.radians(optimal_rotation)
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

    print(
        f"Optimal parameters image: Found {len(x_coords_px)} valid cells within image bounds"
    )
    print(
        f"Optimal parameters coordinate ranges: x=[{x_coords_px.min()}-{x_coords_px.max()}], y=[{y_coords_px.min()}-{y_coords_px.max()}]"
    )

    # Create a transparent overlay
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw circles for each cell position
    radius = 2  # Small radius for cell markers
    for x, y in zip(x_coords_px, y_coords_px):
        # Draw semi-transparent green circle
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(0, 255, 0, 128),  # Semi-transparent green
        )

    # Convert base image to RGBA if needed
    if image.mode != "RGBA":
        image_copy = image.convert("RGBA")
    else:
        image_copy = image.copy()

    # Composite the overlay onto the image
    result_image = Image.alpha_composite(image_copy, overlay)

    # Convert back to RGB for PNG saving
    if result_image.mode == "RGBA":
        rgb_image = Image.new("RGB", result_image.size, (255, 255, 255))
        rgb_image.paste(result_image, mask=result_image.split()[-1])
        result_image = rgb_image

    # Save the image
    result_image.save(output_path, "PNG")
    print(f"Saved optimal parameters image to: {output_path}")


if __name__ == "__main__":

    if not os.path.exists(svs_file):
        print(f"Error: SVS file not found: {svs_file}")
        exit(1)

    print(f"Processing: {svs_file}")
    extract_second_scale(svs_file, output_file, h5ad_file)
