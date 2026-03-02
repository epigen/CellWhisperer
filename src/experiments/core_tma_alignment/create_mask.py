import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Read the h5ad file
print("Loading h5ad file...")
adata = sc.read_h5ad("TMA1.h5ad")

# Filter for core A-3
core_a3 = adata[adata.obs["coreID"] == "A-3"]
print(f"✓ Found {core_a3.n_obs} cells with coreID='A-3'")

# Get coordinates in mm
x_mm = core_a3.obs["x_slide_mm"].values
y_mm = core_a3.obs["y_slide_mm"].values

# FULL SLIDE IMAGE DIMENSIONS
full_img_width = 49800  # TODO extract this from the svs file
full_img_height = 49837

# Alignment parameters - using same scale factor for both dimensions
scale_factor = 0.7958
rotation_degrees = 270.0

print(f"\nUsing uniform scale factor: {scale_factor}")

# Step 1: Convert mm to pixel coordinates
mm_to_pixel_x = (full_img_width / 10.0) * scale_factor
mm_to_pixel_y = (full_img_height / 10.0) * scale_factor

print(f"mm_to_pixel_x: {mm_to_pixel_x:.4f}")
print(f"mm_to_pixel_y: {mm_to_pixel_y:.4f}")

x_coords_px = x_mm * mm_to_pixel_x
y_coords_px = y_mm * mm_to_pixel_y


# Step 3: Apply rotation
center_x = full_img_width / 2
center_y = full_img_height / 2

angle_rad = np.radians(rotation_degrees)
cos_angle = np.cos(angle_rad)
sin_angle = np.sin(angle_rad)

x_centered = x_coords_px - center_x
y_centered = y_coords_px - center_y

x_rotated = x_centered * cos_angle - y_centered * sin_angle
y_rotated = x_centered * sin_angle + y_centered * cos_angle

x_pixels_full = x_rotated + center_x
y_pixels_full = y_rotated + center_y

# Step 4: Offset to patch origin
x_min_full = x_pixels_full.min()
y_min_full = y_pixels_full.min()

x_pixels = x_pixels_full - x_min_full
y_pixels = y_pixels_full - y_min_full

# Determine mask size based on coordinate span
mask_width = int(np.ceil(x_pixels.max())) + 1
mask_height = int(np.ceil(y_pixels.max())) + 1

print(f"\nCreating mask of size: {mask_width} x {mask_height}")

# Create a blank RGBA mask (transparent background)
mask = np.zeros((mask_height, mask_width, 4), dtype=np.uint8)

# Draw red dots at cell positions with full opacity
dot_radius = 3  # radius in pixels for each dot
for x, y in zip(x_pixels, y_pixels):
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
mask_img.save("TMA1_A-3_mask.png")
print(f"✓ Mask saved as 'TMA1_A-3_mask.png'")

# Also create a visualization with matplotlib for comparison
fig, ax = plt.subplots(1, 1, figsize=(14, 13))
ax.imshow(mask, cmap="gray")
ax.set_title(
    f"Core A-3 Cell Mask: {core_a3.n_obs} cells\n"
    + f"Size: {mask_width} x {mask_height} pixels",
    fontsize=14,
)
ax.set_xlabel("X (pixels)", fontsize=12)
ax.set_ylabel("Y (pixels)", fontsize=12)
plt.tight_layout()
plt.savefig("TMA1_A-3_mask_visualization.png", dpi=150, bbox_inches="tight")
print(f"✓ Visualization saved as 'TMA1_A-3_mask_visualization.png'")

print(f"\n" + "=" * 60)
print(f"MASK SUMMARY:")
print("=" * 60)
print(f"  Cells: {core_a3.n_obs}")
print(f"  Mask size: {mask_width} x {mask_height} pixels")
print(f"  Dot radius: {dot_radius} pixels")
print(f"  Scale factor (uniform): {scale_factor}")
print(f"  Format: RGBA (transparent background, red dots)")
print(f"  Non-transparent pixels: {np.sum(mask[:, :, 3] > 0)}")
print("=" * 60)
