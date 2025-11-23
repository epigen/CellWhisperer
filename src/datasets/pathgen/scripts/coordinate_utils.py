"""
Coordinate utility functions for PathGen dataset processing.

PathGen uses center-oriented coordinates as the primary system:
- x_pixel, y_pixel: Center coordinates of 672x672 patches

Physical specifications for TCGA WSIs:
- Magnification: 40x objective power  
- Pixel size: 0.25 μm/pixel
- Patch size: 672x672 pixels = 168 μm × 168 μm physical area
"""

from typing import Tuple, Union


def center_to_topleft(center_x: Union[int, float], 
                     center_y: Union[int, float], 
                     patch_size: int = 672) -> Tuple[int, int]:
    """
    Convert center coordinates to top-left coordinates for OpenSlide.
    
    Args:
        center_x: Center x coordinate
        center_y: Center y coordinate 
        patch_size: Size of the patch (default 672 for PathGen)
        
    Returns:
        Tuple of (topleft_x, topleft_y) coordinates
        
    Example:
        >>> center_to_topleft(400, 300, 672)
        (64, -36)
    """
    half_patch = patch_size // 2
    return int(center_x - half_patch), int(center_y - half_patch)


def topleft_to_center(topleft_x: Union[int, float], 
                     topleft_y: Union[int, float], 
                     patch_size: int = 672) -> Tuple[int, int]:
    """
    Convert top-left coordinates to center coordinates.
    
    Args:
        topleft_x: Top-left x coordinate
        topleft_y: Top-left y coordinate
        patch_size: Size of the patch (default 672 for PathGen)
        
    Returns:
        Tuple of (center_x, center_y) coordinates
        
    Example:
        >>> topleft_to_center(64, -36, 672)
        (400, 300)
    """
    half_patch = patch_size // 2
    return int(topleft_x + half_patch), int(topleft_y + half_patch)


def read_patch_from_center(slide, center_x: int, center_y: int, patch_size: int = 672):
    """
    Read a patch from a WSI using center coordinates.
    
    Args:
        slide: OpenSlide slide object
        center_x: Center x coordinate
        center_y: Center y coordinate
        patch_size: Size of the patch to extract
        
    Returns:
        PIL Image patch
        
    Example:
        >>> import openslide
        >>> slide = openslide.OpenSlide("path/to/slide.svs")
        >>> patch = read_patch_from_center(slide, 1000, 1000)
    """
    topleft_x, topleft_y = center_to_topleft(center_x, center_y, patch_size)
    
    # OpenSlide expects top-left coordinates
    patch = slide.read_region(
        (topleft_x, topleft_y), 
        level=0, 
        size=(patch_size, patch_size)
    )
    return patch


def get_physical_coordinates(center_x: Union[int, float], center_y: Union[int, float], pixel_size: float = 0.25) -> Tuple[float, float]:
    """
    Convert pixel coordinates to physical coordinates in micrometers.
    
    Args:
        center_x: Center x coordinate in pixels
        center_y: Center y coordinate in pixels
        pixel_size: Pixel size in μm/pixel (default 0.25 for TCGA 40x)
        
    Returns:
        Tuple of (x_um, y_um) physical coordinates in micrometers
        
    Example:
        >>> x_um, y_um = get_physical_coordinates(1000, 1000)
        >>> # With 0.25 μm/pixel: (250.0, 250.0)
    """
    return float(center_x * pixel_size), float(center_y * pixel_size)