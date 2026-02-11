"""Image processing utilities for lymphoma CosMx data processing."""

import numpy as np
from PIL import Image
import openslide
import logging

logger = logging.getLogger(__name__)


class OpenSlideWrapper:
    """Wrapper class to provide a consistent interface for openslide images at full resolution."""

    def __init__(self, slide_path):
        self.slide = openslide.OpenSlide(slide_path)

        # Log slide information
        logger.info(f"Slide has {self.slide.level_count} levels")
        logger.info(f"Level dimensions: {self.slide.level_dimensions}")
        logger.info(f"Level downsamples: {self.slide.level_downsamples}")
        logger.info(f"Using full resolution level 0, dimensions: {self.slide.level_dimensions[0]}")

    @property
    def size(self):
        """Return (width, height) at full resolution."""
        return self.slide.level_dimensions[0]

    @property
    def level_count(self):
        """Return number of levels in the slide."""
        return self.slide.level_count

    @property
    def level_dimensions(self):
        """Return dimensions for all levels."""
        return self.slide.level_dimensions

    @property
    def level_downsamples(self):
        """Return downsample factors for all levels."""
        return self.slide.level_downsamples

    def read_region(self, location, size):
        """Read a region from the slide at full resolution (level 0)."""
        # Read region at level 0 (full resolution)
        image = self.slide.read_region(location, 0, size)

        # Convert RGBA to RGB if needed
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background

        return image

    def close(self):
        """Close the slide."""
        self.slide.close()


def handle_magnification(adata, slide_wrapper):
    """Handle magnification and scaling for the image.

    Returns the slide wrapper (always uses full resolution).
    """
    width, height = slide_wrapper.size
    logger.info(f"Image dimensions at full resolution: {width}x{height}")
    return slide_wrapper


def crop_tile(slide_wrapper, x, y, size):
    """Crops a tile from an OpenSlide image."""
    # Ensure coordinates are within image bounds
    img_width, img_height = slide_wrapper.size
    x = max(0, min(x, img_width - size))
    y = max(0, min(y, img_height - size))

    # Read the region from the slide
    tile_image = slide_wrapper.read_region((x, y), (size, size))
    return np.array(tile_image)


def prepare_visualization_image(slide_wrapper, target_max_dim=2000):
    """Prepare image for visualization by downsampling if needed."""
    width, height = slide_wrapper.size
    downsample = max(width, height) / target_max_dim

    if downsample > 1:
        # Calculate new dimensions
        new_width = int(width / downsample)
        new_height = int(height / downsample)

        # Find the best level for our target downsample
        best_level = slide_wrapper.slide.get_best_level_for_downsample(downsample)

        # Read the entire slide at the best level
        level_dims = slide_wrapper.slide.level_dimensions[best_level]
        img_pil = slide_wrapper.slide.read_region((0, 0), best_level, level_dims)

        # Convert RGBA to RGB if needed
        if img_pil.mode == "RGBA":
            background = Image.new("RGB", img_pil.size, (255, 255, 255))
            background.paste(img_pil, mask=img_pil.split()[-1])
            img_pil = background

        # Resize to exact target dimensions if needed
        if img_pil.size != (new_width, new_height):
            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)

        scale_factor_img = 1.0 / downsample
    else:
        # Read the entire slide at full resolution
        img_pil = slide_wrapper.read_region((0, 0), slide_wrapper.size)
        scale_factor_img = 1.0

    img_np = np.array(img_pil)[:, :, :3]  # Keep RGB, drop alpha if present
    return img_np, scale_factor_img
