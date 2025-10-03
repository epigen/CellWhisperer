#!/usr/bin/env python3
"""
Script to open an SVS file and save a 10x reduced version as PNG.
"""

import openslide
from PIL import Image
import os

def reduce_svs_image(input_path, output_path, reduction_factor=10):
    """
    Open an SVS file and save a reduced version as PNG.
    
    Args:
        input_path: Path to the input SVS file
        output_path: Path for the output PNG file
        reduction_factor: Factor by which to reduce the image size (default: 10)
    """
    # Open the SVS file
    slide = openslide.OpenSlide(input_path)
    
    # Get the dimensions of the slide at level 0 (highest resolution)
    width, height = slide.dimensions
    
    # Calculate new dimensions
    new_width = width // reduction_factor
    new_height = height // reduction_factor
    
    print(f"Original dimensions: {width} x {height}")
    print(f"Reduced dimensions: {new_width} x {new_height}")
    print(f"Available levels: {slide.level_count}")
    print(f"Level dimensions: {slide.level_dimensions}")
    print(f"Level downsamples: {slide.level_downsamples}")
    
    # Find the best level for our desired downsample factor
    best_level = slide.get_best_level_for_downsample(reduction_factor)
    level_downsample = slide.level_downsamples[best_level]
    level_width, level_height = slide.level_dimensions[best_level]
    
    print(f"Using level {best_level} with downsample factor {level_downsample}")
    print(f"Level {best_level} dimensions: {level_width} x {level_height}")
    
    # Read the region at the selected level
    image = slide.read_region((0, 0), best_level, (level_width, level_height))
    
    # Convert RGBA to RGB if needed
    if image.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
        image = background
    
    # If the level downsample doesn't exactly match our target, resize to get exact dimensions
    if level_downsample != reduction_factor:
        additional_scale = level_downsample / reduction_factor
        final_width = int(level_width / additional_scale)
        final_height = int(level_height / additional_scale)
        print(f"Additional resize needed: {level_width}x{level_height} -> {final_width}x{final_height}")
        image = image.resize((final_width, final_height), Image.Resampling.LANCZOS)
    
    # Save as PNG
    image.save(output_path, 'PNG')
    
    # Close the slide
    slide.close()
    
    print(f"Saved reduced image to: {output_path}")

if __name__ == "__main__":
    input_file = "/home/moritz/code/cellwhisperer/resources/lymphoma_cosmx_small/CosMx_1K_CART_1__61360.svs"
    output_file = "/home/moritz/code/cellwhisperer/resources/lymphoma_cosmx_small/CosMx_1K_CART_1__61360_reduced.png"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        exit(1)
    
    reduce_svs_image(input_file, output_file, reduction_factor=10)
