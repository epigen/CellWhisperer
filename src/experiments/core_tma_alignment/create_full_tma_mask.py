import scanpy as sc
import numpy as np
from PIL import Image

# Read the h5ad file
print("Loading h5ad file...")
adata = sc.read_h5ad("TMA1.h5ad")

# Use ALL TMA1 cells (no core filter)
tma1_cells = adata[adata.obs["tmaID"] == "TMA1"]
print(f"✓ Found {tma1_cells.n_obs} cells in TMA1 (all cores)")

# Get coordinates in mm
x_mm = tma1_cells.obs["x_slide_mm"].values
y_mm = tma1_cells.obs["y_slide_mm"].values

print(f"\nFull TMA1 coordinate ranges (mm):")
print(f"  X: [{x_mm.min():.4f}, {x_mm.max():.4f}]")
print(f"  Y: [{y_mm.min():.4f}, {y_mm.max():.4f}]")

# FULL SLIDE IMAGE DIMENSIONS. Setting fixed here as a scalar, compatible with /home/moritz/code/cellwhisperer/src/datasets/lymphoma_cosmx_large/scripts/process_data.py
full_img_width = 49800
full_img_height = 49800

# Alignment parameters - using 1/10th resolution
scale_factor = 0.7958 * 0.1
x_offset_pixels = 0
y_offset_pixels = 0
rotation_degrees = 270.0

print(f"\nUsing 1/10th resolution scale factor: {scale_factor}")

# Step 1: Convert mm to pixel coordinates
mm_to_pixel_x = (full_img_width / 10.0) * scale_factor
mm_to_pixel_y = (full_img_height / 10.0) * scale_factor

print(f"mm_to_pixel_x: {mm_to_pixel_x:.4f}")
print(f"mm_to_pixel_y: {mm_to_pixel_y:.4f}")

x_coords_px = x_mm * mm_to_pixel_x
y_coords_px = y_mm * mm_to_pixel_y

# Step 2: Apply offsets (also scaled down by 1/10)
x_coords_px += x_offset_pixels * 0.1
y_coords_px += y_offset_pixels * 0.1

print(f"\nAfter scaling and offsets:")
print(f"  X range: [{x_coords_px.min():.2f}, {x_coords_px.max():.2f}]")
print(f"  Y range: [{y_coords_px.min():.2f}, {y_coords_px.max():.2f}]")

# Step 3: Apply rotation around scaled slide center
center_x = (full_img_width / 10.0) / 2
center_y = (full_img_height / 10.0) / 2

angle_rad = np.radians(rotation_degrees)
cos_angle = np.cos(angle_rad)
sin_angle = np.sin(angle_rad)

print(
    f"\nApplying {rotation_degrees}° rotation around center ({center_x:.1f}, {center_y:.1f})"
)

x_centered = x_coords_px - center_x
y_centered = y_coords_px - center_y

x_rotated = x_centered * cos_angle - y_centered * sin_angle
y_rotated = x_centered * sin_angle + y_centered * cos_angle

x_pixels_full = x_rotated + center_x
y_pixels_full = y_rotated + center_y

print(f"\nAfter rotation (final coordinates):")
print(f"  X range: [{x_pixels_full.min():.2f}, {x_pixels_full.max():.2f}]")
print(f"  Y range: [{y_pixels_full.min():.2f}, {y_pixels_full.max():.2f}]")

# NO OFFSET - keep coordinates as is, filter out negatives
# Use the rotated coordinates directly
x_pixels = x_pixels_full
y_pixels = y_pixels_full

# Filter out cells with negative coordinates
valid_mask = (x_pixels >= 0) & (y_pixels >= 0)
x_pixels_valid = x_pixels[valid_mask]
y_pixels_valid = y_pixels[valid_mask]

print(f"\n" + "=" * 60)
print(f"Filtering negative coordinates:")
print("=" * 60)
print(f"  Total cells before filtering: {len(x_pixels)}")
print(f"  Cells with negative X: {np.sum(x_pixels < 0)}")
print(f"  Cells with negative Y: {np.sum(y_pixels < 0)}")
print(f"  Valid cells (both X,Y >= 0): {len(x_pixels_valid)}")

# Determine mask size based on maximum coordinates (not shifted)
mask_width = int(np.ceil(x_pixels_valid.max())) + 1 if len(x_pixels_valid) > 0 else 1
mask_height = int(np.ceil(y_pixels_valid.max())) + 1 if len(y_pixels_valid) > 0 else 1

print(f"\n" + "=" * 60)
print(f"Creating full TMA1 mask at 1/10th resolution")
print("=" * 60)
print(f"  Mask size: {mask_width} x {mask_height} pixels")
print(f"  Valid cells to plot: {len(x_pixels_valid)}")

# Create a blank RGBA mask (transparent background)
mask = np.zeros((mask_height, mask_width, 4), dtype=np.uint8)

# Draw red dots at cell positions
dot_radius = 1  # Smaller radius for 1/10th resolution
print(f"  Dot radius: {dot_radius} pixel")

for x, y in zip(x_pixels_valid, y_pixels_valid):
    x_int = int(round(x))
    y_int = int(round(y))

    # Draw a filled circle for each cell
    for dy in range(-dot_radius, dot_radius + 1):
        for dx in range(-dot_radius, dot_radius + 1):
            if dx * dx + dy * dy <= dot_radius * dot_radius:
                px = x_int + dx
                py = y_int + dy
                if 0 <= px < mask_width and 0 <= py < mask_height:
                    mask[py, px] = [255, 0, 0, 255]  # Red with full opacity (RGBA)

# Save the mask with transparency
mask_img = Image.fromarray(mask, mode="RGBA")
mask_img.save("TMA1_full_mask_0.1x.png")
print(f"\n✓ Mask saved as 'TMA1_full_mask_0.1x.png'")

print(f"\n" + "=" * 60)
print(f"FULL TMA1 MASK SUMMARY:")
print("=" * 60)
print(f"  Total cells in h5ad: {tma1_cells.n_obs}")
print(f"  Cells plotted (valid coordinates): {len(x_pixels_valid)}")
print(f"  Cells excluded (negative coords): {tma1_cells.n_obs - len(x_pixels_valid)}")
print(f"  Mask size: {mask_width} x {mask_height} pixels")
print(f"  Resolution scale: 0.1x (1/10th)")
print(f"  Scale factor: {scale_factor}")
print(f"  Dot radius: {dot_radius} pixel")
print(f"  Format: RGBA (transparent background, red dots)")
print(f"  Non-transparent pixels: {np.sum(mask[:, :, 3] > 0)}")
print(f"  Full resolution equivalent: {mask_width*10} x {mask_height*10} pixels")
print(f"  NO OFFSET APPLIED - coordinates preserved exactly")
print("=" * 60)
