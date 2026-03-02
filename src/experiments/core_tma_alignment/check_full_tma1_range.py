import scanpy as sc
import numpy as np

# Read the h5ad file
adata = sc.read_h5ad('TMA1.h5ad')

# Filter for ALL TMA1 cells (all cores)
tma1_cells = adata[adata.obs['tmaID'] == 'TMA1']
print(f"Total cells in TMA1: {tma1_cells.n_obs}")

# Get mm coordinates
x_mm = tma1_cells.obs['x_slide_mm'].values
y_mm = tma1_cells.obs['y_slide_mm'].values

print(f"\nFull TMA1 coordinate ranges (mm):")
print(f"  X: [{x_mm.min():.4f}, {x_mm.max():.4f}] - span: {x_mm.max()-x_mm.min():.4f}")
print(f"  Y: [{y_mm.min():.4f}, {y_mm.max():.4f}] - span: {y_mm.max()-y_mm.min():.4f}")

# Apply the same transformation
img_width = 3186  # from TMA1_A-3.tiff (but this is just ONE core!)
img_height = 3041

scale_factor_x = 0.6651145121
scale_factor_y = 0.7046553145
x_offset_pixels = -484
y_offset_pixels = -1046

mm_to_pixel_x = (img_width / 10.0) * scale_factor_x
mm_to_pixel_y = (img_height / 10.0) * scale_factor_y

x_coords_px = x_mm * mm_to_pixel_x + x_offset_pixels
y_coords_px = y_mm * mm_to_pixel_y + y_offset_pixels

print(f"\nFull TMA1 coordinate ranges (pixels, before rotation):")
print(f"  X: [{x_coords_px.min():.2f}, {x_coords_px.max():.2f}] - span: {x_coords_px.max()-x_coords_px.min():.2f}")
print(f"  Y: [{y_coords_px.min():.2f}, {y_coords_px.max():.2f}] - span: {y_coords_px.max()-y_coords_px.min():.2f}")

print(f"\n** The TIFF image size ({img_width} x {img_height}) is just for ONE core (A-3)")
print(f"** But the coordinates are for the ENTIRE TMA1 slide!")
print(f"** You need the full TMA1 slide image, not individual core patches")
