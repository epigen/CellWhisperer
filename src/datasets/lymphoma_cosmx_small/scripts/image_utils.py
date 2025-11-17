"""Image processing utilities for lymphoma CosMx data processing."""

import numpy as np
from PIL import Image
import openslide
import logging

logger = logging.getLogger(__name__)


class OpenSlideWrapper:
    """Wrapper class to provide a consistent interface for openslide and PIL images."""

    def __init__(self, slide_path, dataset_name):
        self.slide = openslide.OpenSlide(slide_path)
        self.dataset_name = dataset_name
        self._level = 0  # Default to highest resolution
        self._downsample_factor = 1.0

        # Log slide information
        logger.info(f"Slide has {self.slide.level_count} levels")
        logger.info(f"Level dimensions: {self.slide.level_dimensions}")
        logger.info(f"Level downsamples: {self.slide.level_downsamples}")

        # Handle magnification scaling
        if "detailed" not in dataset_name and "singlecell" not in dataset_name:
            # For standard resolution, use level 1 if available (2x downsampled)
            if self.slide.level_count > 1:
                self._level = 1
                self._downsample_factor = self.slide.level_downsamples[1]
                logger.info(
                    f"Standard variant: Using level {self._level} with downsample factor {self._downsample_factor:.2f}"
                )
                logger.info(
                    f"Standard variant: Level {self._level} dimensions: {self.slide.level_dimensions[self._level]}"
                )
            else:
                logger.info("Only one level available, using level 0")
        else:
            # For detailed variant, explicitly use level 0 (full resolution)
            self._level = 0
            self._downsample_factor = 1.0
            logger.info(
                f"Detailed variant: Using full resolution level {self._level} (no downsampling)"
            )
            logger.info(
                f"Detailed variant: Level {self._level} dimensions: {self.slide.level_dimensions[self._level]}"
            )

    @property
    def size(self):
        """Return (width, height) of the current level."""
        return self.slide.level_dimensions[self._level]

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
        """Read a region from the slide at the current level."""
        # Convert location from current level to level 0 coordinates
        level_0_location = (
            int(location[0] * self._downsample_factor),
            int(location[1] * self._downsample_factor),
        )

        # Read region at current level
        image = self.slide.read_region(level_0_location, self._level, size)

        # Convert RGBA to RGB if needed
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background

        return image

    def close(self):
        """Close the slide."""
        self.slide.close()


def handle_magnification(adata, slide_wrapper, dataset_name):
    """Handle magnification and scaling for the image.

    Returns the slide wrapper (no changes needed as scaling is handled in wrapper).
    """
    width, height = slide_wrapper.size
    logger.info(f"Image dimensions after magnification handling: {width}x{height}")

    # Update pixel size metadata if available
    if "pixel_size" in adata.uns:
        pixel_size = adata.uns["pixel_size"]
        if "detailed" not in dataset_name and slide_wrapper._level > 0:
            # Update pixel size to reflect the downsampling
            adata.uns["pixel_size"] = pixel_size * slide_wrapper._downsample_factor
            logger.info(f"Updated pixel size to {adata.uns['pixel_size']:.3f} μm/pixel")

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

        # Read a downsampled version of the entire slide
        # Find the best level for our target downsample
        total_downsample = downsample * slide_wrapper._downsample_factor
        best_level = slide_wrapper.slide.get_best_level_for_downsample(total_downsample)

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
        # Read the entire slide at current level
        img_pil = slide_wrapper.read_region((0, 0), slide_wrapper.size)
        scale_factor_img = 1.0

    img_np = np.array(img_pil)[:, :, :3]  # Keep RGB, drop alpha if present
    return img_np, scale_factor_img
