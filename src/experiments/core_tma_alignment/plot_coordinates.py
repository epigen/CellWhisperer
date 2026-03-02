import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile

# Read the h5ad file
print("Loading h5ad file...")
adata = sc.read_h5ad('TMA1.h5ad')

# Check for core A-1
print(f"\nUnique coreID values (first 20): {sorted(adata.obs['coreID'].unique())[:20]}")
print(f"\nTotal unique cores: {len(adata.obs['coreID'].unique())}")

# Filter for core A-1
core_a1 = adata[adata.obs['coreID'] == 'A-1']
print(f"\nCells in core A-1: {core_a1.n_obs}")

if core_a1.n_obs == 0:
    print("\nNo cells found for core A-1. Let me check other possible column names...")
    if 'TMA_coreID' in adata.obs.columns:
        print(f"TMA_coreID unique values (first 20): {sorted(adata.obs['TMA_coreID'].unique())[:20]}")
else:
    # Get coordinates
    # Check which coordinate columns are available
    coord_cols = [col for col in core_a1.obs.columns if any(x in col.lower() for x in ['x_', 'y_', '_x', '_y'])]
    print(f"\nAvailable coordinate columns: {coord_cols}")
    
    # Use x_slide_mm_transform and y_slide_mm_transform
    if 'x_slide_mm_transform' in core_a1.obs.columns and 'y_slide_mm_transform' in core_a1.obs.columns:
        x_coords = core_a1.obs['x_slide_mm_transform'].values
        y_coords = core_a1.obs['y_slide_mm_transform'].values
        print(f"\nCoordinates in mm (transformed):")
        print(f"  x range: [{x_coords.min():.4f}, {x_coords.max():.4f}] mm")
        print(f"  y range: [{y_coords.min():.4f}, {y_coords.max():.4f}] mm")
        print(f"  x span: {x_coords.max() - x_coords.min():.4f} mm")
        print(f"  y span: {y_coords.max() - y_coords.min():.4f} mm")
        
    # Also check FOV pixel coordinates
    if 'x_FOV_px' in core_a1.obs.columns and 'y_FOV_px' in core_a1.obs.columns:
        x_fov = core_a1.obs['x_FOV_px'].values
        y_fov = core_a1.obs['y_FOV_px'].values
        print(f"\nCoordinates in FOV pixels:")
        print(f"  x range: [{x_fov.min():.2f}, {x_fov.max():.2f}] px")
        print(f"  y range: [{y_fov.min():.2f}, {y_fov.max():.2f}] px")
