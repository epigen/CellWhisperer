import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import tifffile

# Read the h5ad file
print("Loading h5ad file...")
adata = sc.read_h5ad('TMA1.h5ad')

# Load the TIFF image
print("Loading TIFF image...")
img = tifffile.imread('TMA1_A-3.tiff')
img_height, img_width = img.shape[:2]
print(f"Image shape: {img.shape} (H x W x C)")
print(f"Image dimensions: {img_width} x {img_height} pixels")

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

# From alignment.csv for TMA1
scale_factor_x = 0.6651145121
scale_factor_y = 0.7046553145
x_offset_pixels = -484
y_offset_pixels = -1046

print(f"\nAlignment parameters from CSV:")
print(f"  scale_factor_x: {scale_factor_x}")
print(f"  scale_factor_y: {scale_factor_y}")
print(f"  x_offset_pixels: {x_offset_pixels}")
print(f"  y_offset_pixels: {y_offset_pixels}")

# Step 1: Convert mm to pixel coordinates
mm_to_pixel_x = (img_width / 10.0) * scale_factor_x
mm_to_pixel_y = (img_height / 10.0) * scale_factor_y

print(f"\nStep 1: MM to pixel conversion")
print(f"  mm_to_pixel_x: {mm_to_pixel_x:.4f}")
print(f"  mm_to_pixel_y: {mm_to_pixel_y:.4f}")

x_coords_px = x_mm * mm_to_pixel_x
y_coords_px = y_mm * mm_to_pixel_y

print(f"  X range (px): [{x_coords_px.min():.2f}, {x_coords_px.max():.2f}]")
print(f"  Y range (px): [{y_coords_px.min():.2f}, {y_coords_px.max():.2f}]")

# Step 2: Apply offsets
print(f"\nStep 2: Apply offsets")
x_coords_px += x_offset_pixels
y_coords_px += y_offset_pixels

print(f"  X range (px): [{x_coords_px.min():.2f}, {x_coords_px.max():.2f}]")
print(f"  Y range (px): [{y_coords_px.min():.2f}, {y_coords_px.max():.2f}]")

# Step 3: Apply rotation
rotation_degrees = 270.0
center_x = img_width / 2
center_y = img_height / 2

angle_rad = np.radians(rotation_degrees)
cos_angle = np.cos(angle_rad)
sin_angle = np.sin(angle_rad)

print(f"\nStep 3: Apply rotation ({rotation_degrees}° around center)")
print(f"  Center: ({center_x:.1f}, {center_y:.1f})")

x_centered = x_coords_px - center_x
y_centered = y_coords_px - center_y

x_rotated = x_centered * cos_angle - y_centered * sin_angle
y_rotated = x_centered * sin_angle + y_centered * cos_angle

x_pixels = x_rotated + center_x
y_pixels = y_rotated + center_y

print(f"  X range (px): [{x_pixels.min():.2f}, {x_pixels.max():.2f}]")
print(f"  Y range (px): [{y_pixels.min():.2f}, {y_pixels.max():.2f}]")

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(14, 13))
ax.imshow(img)
ax.scatter(x_pixels, y_pixels, c='red', s=6, alpha=0.6, edgecolors='none')
ax.set_title(f'Core A-3: {core_a3.n_obs} cells overlaid on TMA1_A-3.tiff\n' + 
             f'With offsets ({x_offset_pixels}, {y_offset_pixels}), scale ({scale_factor_x:.4f}, {scale_factor_y:.4f}), 270° rotation', 
             fontsize=14)
ax.set_xlabel('X (pixels)', fontsize=12)
ax.set_ylabel('Y (pixels)', fontsize=12)
ax.set_xlim(0, img_width)
ax.set_ylim(img_height, 0)
plt.tight_layout()
plt.savefig('TMA1_A-3_overlay_with_offsets.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved as 'TMA1_A-3_overlay_with_offsets.png'")
