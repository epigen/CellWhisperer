"""Coordinate transformation utilities for lymphoma CosMx data processing."""

import numpy as np
import logging
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import subprocess
from pathlib import Path

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


def _evaluate_offset_batch(args):
    """
    Evaluate a batch of offset positions for a single core (parallelizable unit).
    
    Args:
        args: Tuple containing (core_coords_relative, value_channel, batch_offsets)
    
    Returns:
        List of tuples: [((dx, dy), normalized_score, valid_cells), ...]
    """
    core_coords_relative, value_channel, batch_offsets = args
    
    results = []
    for dx, dy in batch_offsets:
        # Apply offset to coordinates
        offset_coords = core_coords_relative + np.array([dx, dy])
        
        # Check bounds and calculate score
        score = 0
        valid_cells = 0
        
        for cell_x, cell_y in offset_coords:
            cell_x_int, cell_y_int = int(round(cell_x)), int(round(cell_y))
            
            # Check if coordinates are within the image bounds
            if (0 <= cell_x_int < value_channel.shape[1] and 
                0 <= cell_y_int < value_channel.shape[0]):
                score += value_channel[cell_y_int, cell_x_int]
                valid_cells += 1
        
        # Normalize score by number of valid cells
        normalized_score = score / valid_cells if valid_cells > 0 else np.inf  # inf for invalid positions
        results.append(((dx, dy), normalized_score, valid_cells))
    
    return results


def perform_core_level_alignment(adata_hr, slide_wrapper, spatial_coords, dataset_name, n_jobs=None, batch_size=200):
    """
    Perform core-level alignment using HSV value channel optimization with inner parallelization.
    
    For each core:
    1. Get bounding box of all cells in that core (+50px margin)
    2. Extract level-0 image patch for that core 
    3. Convert to HSV and use only value channel
    4. Run 101x101 offset grid search (-50 to +50 pixels) in parallel
    5. For each offset, sum value channel at cell positions
    6. Choose offset with MINIMUM summed value (dark regions)
    7. Apply offset to all cells in that core
    
    Args:
        adata_hr: AnnData object with core_id information
        slide_wrapper: OpenSlideWrapper for accessing image data
        spatial_coords: Current spatial coordinates [x, y] for all cells
        dataset_name: Name of the dataset (for creating new slide wrappers)
        n_jobs: Number of parallel jobs for inner grid search (default: cpu_count())
        batch_size: Number of offset positions to evaluate per batch (default: 200)
    
    Returns:
        numpy array: Aligned spatial coordinates
    """
    logger.info("Starting core-level alignment optimization (inner parallelization)")
    
    # Check if we have core information
    if 'core_id' not in adata_hr.obs.columns:
        logger.warning("No core_id information found, skipping core-level alignment")
        return spatial_coords
        
    # Copy coordinates for modification
    aligned_coords = spatial_coords.copy()
    
    # Get unique cores
    cores = adata_hr.obs['core_id'].dropna().unique()
    logger.info(f"Found {len(cores)} cores for alignment: {sorted(cores)}")
    
    if n_jobs is None:
        n_jobs = cpu_count()
    logger.info(f"Using inner parallelization with {n_jobs} workers, batch size {batch_size}")
    
    for core in cores:
        logger.info(f"\nProcessing core {core} for alignment optimization...")
        
        # Get cells for this core
        core_mask = adata_hr.obs['core_id'] == core
        core_coords = spatial_coords[core_mask]
        
        if len(core_coords) == 0:
            logger.warning(f"No cells found for core {core}, skipping")
            continue
            
        logger.info(f"Core {core} has {len(core_coords)} cells")
        
        # Get bounding box with 50px margin
        x_min, y_min = core_coords.min(axis=0)
        x_max, y_max = core_coords.max(axis=0)
        
        margin = 50
        bbox_x_min = max(0, int(x_min - margin))
        bbox_y_min = max(0, int(y_min - margin))
        bbox_x_max = min(slide_wrapper.size[0], int(x_max + margin))
        bbox_y_max = min(slide_wrapper.size[1], int(y_max + margin))
        
        bbox_width = bbox_x_max - bbox_x_min
        bbox_height = bbox_y_max - bbox_y_min
        
        logger.info(f"Core {core} bounding box: ({bbox_x_min}, {bbox_y_min}) to ({bbox_x_max}, {bbox_y_max}), size: {bbox_width}x{bbox_height}")
        
        # Extract level-0 image patch
        try:
            region = slide_wrapper.read_region((bbox_x_min, bbox_y_min), (bbox_width, bbox_height))
            region_rgb = region.convert('RGB')
            
            # Convert to HSV and extract value channel using PIL
            region_hsv = region_rgb.convert('HSV')
            region_hsv_np = np.array(region_hsv)
            value_channel = region_hsv_np[:, :, 2]  # V channel
            
            logger.info(f"Extracted {value_channel.shape} value channel for core {core}")
            
        except Exception as e:
            logger.error(f"Failed to extract image region for core {core}: {e}")
            continue
        
        # Adjust core coordinates to be relative to bbox
        core_coords_relative = core_coords - np.array([bbox_x_min, bbox_y_min])
        
        # Prepare all offset combinations (-50 to +50 pixels)
        offset_range = range(-50, 51)
        all_offsets = [(dx, dy) for dx in offset_range for dy in offset_range]
        
        logger.info(f"Starting grid search for core {core}: {len(offset_range)}x{len(offset_range)} = {len(all_offsets)} positions")
        
        # Create batches of offsets for parallel processing
        batches = [all_offsets[i:i+batch_size] for i in range(0, len(all_offsets), batch_size)]
        
        # Prepare arguments for parallel processing
        tasks = [(core_coords_relative, value_channel, batch) for batch in batches]
        
        # Process batches in parallel
        all_results = []
        best_score = np.inf  # We're minimizing now
        best_offset = None
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            with tqdm(total=len(all_offsets), desc=f"Core {core} alignment", unit="pos") as pbar:
                futures = {executor.submit(_evaluate_offset_batch, task): len(task[2]) for task in tasks}
                
                for future in as_completed(futures):
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    
                    # Update progress and find current best (minimum)
                    pbar.update(futures[future])
                    
                    # Find best so far for progress display
                    for (dx, dy), score, valid_cells in batch_results:
                        if valid_cells > 0 and score < best_score:
                            best_score = score
                            best_offset = (dx, dy)
                            pbar.set_description(f"Core {core} alignment (best: {dx:+3d},{dy:+3d} = {best_score:.1f})")
        
        # Find overall best result (minimum score)
        if all_results:
            valid_results = [(offset, score, cells) for offset, score, cells in all_results if cells > 0]
            
            if valid_results:
                best_result = min(valid_results, key=lambda x: x[1])  # Minimize score
                best_offset, best_score, valid_cells = best_result
                dx_best, dy_best = best_offset
                
                logger.info(f"Core {core}: Best offset = ({dx_best}, {dy_best}), score = {best_score:.2f}")
                
                # Apply best offset to all cells in this core
                aligned_coords[core_mask] = spatial_coords[core_mask] + np.array([dx_best, dy_best])
                
                # Create visualization
                try:
                    plt.figure(figsize=(10, 8))
                    
                    # Show the value channel as background
                    plt.imshow(value_channel, cmap='gray', alpha=0.7)
                    
                    # Plot original cell positions (relative to bbox) as empty blue circles
                    original_rel = core_coords_relative
                    plt.scatter(original_rel[:, 0], original_rel[:, 1], 
                              facecolors='none', edgecolors='blue', s=10, linewidths=0.5, alpha=0.7, label='Original')
                    
                    # Plot aligned cell positions (relative to bbox) as empty red circles
                    aligned_rel = core_coords_relative + np.array([dx_best, dy_best])
                    plt.scatter(aligned_rel[:, 0], aligned_rel[:, 1], 
                              facecolors='none', edgecolors='red', s=10, linewidths=0.5, alpha=0.9, label='Aligned')
                    
                    plt.title(f'Core {core} Alignment (Minimized)\nOffset: ({dx_best}, {dy_best}), Score: {best_score:.2f}')
                    plt.xlabel('X pixels (relative to bbox)')
                    plt.ylabel('Y pixels (relative to bbox)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Save the plot to results directory
                    project_dir = Path(subprocess.check_output("git rev-parse --show-toplevel", shell=True).decode("utf-8").strip())
                    plot_path = project_dir / f"results/{dataset_name}/core_{core}_alignment.png"
                    # Ensure directory exists
                    plot_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"Saved alignment visualization to {plot_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to create visualization for core {core}: {e}")
                    plt.close()
            else:
                logger.warning(f"No valid alignment found for core {core}")
        else:
            logger.warning(f"No alignment results for core {core}")
    
    logger.info("Core-level alignment optimization completed")
    return aligned_coords
