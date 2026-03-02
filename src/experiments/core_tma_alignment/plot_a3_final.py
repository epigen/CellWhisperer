import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import tifffile

# Read the h5ad file
print("Loading h5ad file...")
adata = sc.read_h5ad('TMA1.h5ad')

# Load the TIFF image (this is a CROPPED core patch, not the full slide)
print("Loading TIFF image...")
img = tifffile.imread('TMA1_A-3.tiff')
patch_height, patch_width = img.shape[:2]
print(f"Core patch image shape: {img.shape} (H x W x C)")
print(f"Core patch dimensions: {patch_width} x {patch_height} pixels")

# Filter for core A-3
core_a3 = adata[adata.obs['coreID'] == 'A-3']
print(f"\n✓ Found {core_a3.n_obs} cells with coreID='A-3'")

# Get coordinates in mm
x_mm = core_a3.obs['x_slide_mm'].values
y_mm = core_a3.obs['y_slide_mm'].values

print(f"\n" + "="*60)
print(f"Coordinate ranges in mm:")
print("="*60)
print(f"  X (mm): [{x_mm.min():.4f}, {x_mm.max():.4f}]")
print(f"  Y (mm): [{y_mm.min():.4f}, {y_mm.max():.4f}]")
print(f"  Span: {x_mm.max()-x_mm.min():.4f} x {y_mm.max()-y_mm.min():.4f} mm")

# FULL SLIDE IMAGE DIMENSIONS (the SVS file)
full_img_width = 49800
full_img_height = 49837

print(f"\n" + "="*60)
print(f"Using FULL slide image dimensions for transformation:")
print("="*60)
print(f"  Full slide: {full_img_width} x {full_img_height} pixels")

# From alignment.csv for TMA1
scale_factor_x = 0.6651145121
scale_factor_y = 0.7046553145
x_offset_pixels = -484
y_offset_pixels = -1046
rotation_degrees = 270.0

print(f"\nAlignment parameters from alignment.csv:")
print(f"  scale_factor_x: {scale_factor_x}")
print(f"  scale_factor_y: {scale_factor_y}")
print(f"  x_offset_pixels: {x_offset_pixels}")
print(f"  y_offset_pixels: {y_offset_pixels}")
print(f"  rotation: {rotation_degrees}°")

# Step 1: Convert mm to pixel coordinates using FULL SLIDE dimensions
mm_to_pixel_x = (full_img_width / 10.0) * scale_factor_x
mm_to_pixel_y = (full_img_height / 10.0) * scale_factor_y

print(f"\n" + "="*60)
print(f"Step 1: MM to pixel conversion")
print("="*60)
print(f"  mm_to_pixel_x: {mm_to_pixel_x:.4f}")
print(f"  mm_to_pixel_y: {mm_to_pixel_y:.4f}")

x_coords_px = x_mm * mm_to_pixel_x
y_coords_px = y_mm * mm_to_pixel_y

print(f"  X range (px): [{x_coords_px.min():.2f}, {x_coords_px.max():.2f}]")
print(f"  Y range (px): [{y_coords_px.min():.2f}, {y_coords_px.max():.2f}]")

# Step 2: Apply pixel offsets
print(f"\nStep 2: Apply offsets")
x_coords_px += x_offset_pixels
y_coords_px += y_offset_pixels

print(f"  X range (px): [{x_coords_px.min():.2f}, {x_coords_px.max():.2f}]")
print(f"  Y range (px): [{y_coords_px.min():.2f}, {y_coords_px.max():.2f}]")

# Step 3: Apply rotation around full slide center
center_x = full_img_width / 2
center_y = full_img_height / 2

angle_rad = np.radians(rotation_degrees)
cos_angle = np.cos(angle_rad)
sin_angle = np.sin(angle_rad)

print(f"\nStep 3: Apply {rotation_degrees}° rotation")
print(f"  Center: ({center_x:.1f}, {center_y:.1f})")

x_centered = x_coords_px - center_x
y_centered = y_coords_px - center_y

x_rotated = x_centered * cos_angle - y_centered * sin_angle
y_rotated = x_centered * sin_angle + y_centered * cos_angle

x_pixels_full = x_rotated + center_x
y_pixels_full = y_rotated + center_y

print(f"  X range (px): [{x_pixels_full.min():.2f}, {x_pixels_full.max():.2f}]")
print(f"  Y range (px): [{y_pixels_full.min():.2f}, {y_pixels_full.max():.2f}]")

# Step 4: Crop to core A-3 patch coordinates
# The TIFF patch is a crop of the full slide - we need to offset coordinates to match
print(f"\n" + "="*60)
print(f"Step 4: Adjust coordinates for cropped core patch (no scaling)")
print("="*60)

# Find bounding box of core A-3 in full slide coordinates
x_min_full = x_pixels_full.min()
y_min_full = y_pixels_full.min()
x_max_full = x_pixels_full.max()
y_max_full = y_pixels_full.max()

print(f"  Core A-3 bounding box in full slide: ")
print(f"    X: [{x_min_full:.2f}, {x_max_full:.2f}]")
print(f"    Y: [{y_min_full:.2f}, {y_max_full:.2f}]")
print(f"    Size: {x_max_full-x_min_full:.2f} x {y_max_full-y_min_full:.2f} px")

# Offset to patch coordinates (relative to crop) - NO SCALING
x_pixels = x_pixels_full - x_min_full
y_pixels = y_pixels_full - y_min_full

print(f"\n  Final coordinates (offset to patch origin, no scaling):")
print(f"    X: [{x_pixels.min():.2f}, {x_pixels.max():.2f}]")
print(f"    Y: [{y_pixels.min():.2f}, {y_pixels.max():.2f}]")

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(14, 13))
ax.imshow(img)
ax.scatter(x_pixels, y_pixels, c='red', s=6, alpha=0.6, edgecolors='none')
ax.set_title(f'Core A-3: {core_a3.n_obs} cells overlaid on TMA1_A-3.tiff\n' + 
             f'Full slide: {full_img_width}x{full_img_height}, Scale: ({scale_factor_x:.4f}, {scale_factor_y:.4f}), 270° rotation', 
             fontsize=12)
ax.set_xlabel('X (pixels)', fontsize=12)
ax.set_ylabel('Y (pixels)', fontsize=12)
ax.set_xlim(0, patch_width)
ax.set_ylim(patch_height, 0)
plt.tight_layout()
plt.savefig('TMA1_A-3_overlay_final.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved as 'TMA1_A-3_overlay_final.png'")

print(f"\n" + "="*60)
print(f"SUMMARY:")
print("="*60)
print(f"  Cells plotted: {core_a3.n_obs}")
print(f"  Full slide dimensions: {full_img_width} x {full_img_height} px")
print(f"  Core patch dimensions: {patch_width} x {patch_height} px")
print(f"  Alignment scale factors: {scale_factor_x:.4f} x {scale_factor_y:.4f}")
print(f"  Coordinate span: {x_max_full-x_min_full:.2f} x {y_max_full-y_min_full:.2f} px")
print("="*60)
