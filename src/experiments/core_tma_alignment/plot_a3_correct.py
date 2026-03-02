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
print("\n" + "="*60)
print("Filtering for core A-3...")
print("="*60)

core_a3 = adata[adata.obs['coreID'] == 'A-3']
print(f"✓ Found {core_a3.n_obs} cells with coreID='A-3'")

# Get coordinates in mm (the correct coordinate system to use)
x_mm = core_a3.obs['x_slide_mm'].values
y_mm = core_a3.obs['y_slide_mm'].values

print(f"\n" + "="*60)
print(f"Coordinate ranges in mm (x_slide_mm, y_slide_mm):")
print("="*60)
print(f"  X (mm): [{x_mm.min():.4f}, {x_mm.max():.4f}]")
print(f"  Y (mm): [{y_mm.min():.4f}, {y_mm.max():.4f}]")
print(f"  X span: {x_mm.max() - x_mm.min():.4f} mm")
print(f"  Y span: {y_mm.max() - y_mm.min():.4f} mm")

# Use the exact logic from convert_mm_to_pixel_coordinates
# Step 1: Convert mm to pixel coordinates with independent X/Y scaling
scale_factor_x = 0.6651145121
scale_factor_y = 0.7046553145

mm_to_pixel_x = (img_width / 10.0) * scale_factor_x
mm_to_pixel_y = (img_height / 10.0) * scale_factor_y

print(f"\n" + "="*60)
print(f"Scaling calculation (following convert_mm_to_pixel_coordinates):")
print("="*60)
print(f"  Image size: {img_width} x {img_height} pixels")
print(f"  scale_factor_x: {scale_factor_x:.10f}")
print(f"  scale_factor_y: {scale_factor_y:.10f}")
print(f"  mm_to_pixel_x: {mm_to_pixel_x:.4f} (= img_width/10 * scale_factor_x)")
print(f"  mm_to_pixel_y: {mm_to_pixel_y:.4f} (= img_height/10 * scale_factor_y)")

x_coords_px = x_mm * mm_to_pixel_x
y_coords_px = y_mm * mm_to_pixel_y

print(f"\nAfter mm to pixel conversion (before offset/rotation):")
print(f"  X (pixels): [{x_coords_px.min():.2f}, {x_coords_px.max():.2f}]")
print(f"  Y (pixels): [{y_coords_px.min():.2f}, {y_coords_px.max():.2f}]")

# Step 2: Apply pixel-based offsets (set to 0 for now, can be adjusted)
x_offset_pixels = 0
y_offset_pixels = 0

print(f"\nApplying pixel offsets: x_offset={x_offset_pixels}, y_offset={y_offset_pixels}")
x_coords_px += x_offset_pixels
y_coords_px += y_offset_pixels

# Step 3: Apply rotation around image center
rotation_degrees = 270.0
center_x = img_width / 2
center_y = img_height / 2

angle_rad = np.radians(rotation_degrees)
cos_angle = np.cos(angle_rad)
sin_angle = np.sin(angle_rad)

print(f"\nApplying {rotation_degrees:.0f}° rotation around center ({center_x:.1f}, {center_y:.1f})")

# Translate to origin, rotate, then translate back
x_centered = x_coords_px - center_x
y_centered = y_coords_px - center_y

x_rotated = x_centered * cos_angle - y_centered * sin_angle
y_rotated = x_centered * sin_angle + y_centered * cos_angle

x_pixels = x_rotated + center_x
y_pixels = y_rotated + center_y

print(f"\nAfter rotation (before offset normalization):")
print(f"  X (pixels): [{x_pixels.min():.2f}, {x_pixels.max():.2f}]")
print(f"  Y (pixels): [{y_pixels.min():.2f}, {y_pixels.max():.2f}]")

# Offset coordinates so minimum is (0, 0)
x_min_offset = x_pixels.min()
y_min_offset = y_pixels.min()

x_pixels = x_pixels - x_min_offset
y_pixels = y_pixels - y_min_offset

print(f"\n" + "="*60)
print(f"After offset normalization to (0,0) - Final pixel coordinates:")
print("="*60)
print(f"  Applied offsets: X={x_min_offset:.2f}, Y={y_min_offset:.2f}")
print(f"  X (pixels): [{x_pixels.min():.2f}, {x_pixels.max():.2f}]")
print(f"  Y (pixels): [{y_pixels.min():.2f}, {y_pixels.max():.2f}]")

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(14, 13))
ax.imshow(img)
ax.scatter(x_pixels, y_pixels, c='red', s=6, alpha=0.6, edgecolors='none')
ax.set_title(f'Core A-3: {core_a3.n_obs} cells overlaid on TMA1_A-3.tiff\n' + 
             f'Using x_slide_mm/y_slide_mm coordinates\n' +
             f'Scale factors: {scale_factor_x:.4f} (X), {scale_factor_y:.4f} (Y), 270° rotation', fontsize=14)
ax.set_xlabel('X (pixels)', fontsize=12)
ax.set_ylabel('Y (pixels)', fontsize=12)
ax.set_xlim(0, img_width)
ax.set_ylim(img_height, 0)  # Invert Y-axis to match image coordinates
plt.tight_layout()
plt.savefig('TMA1_A-3_overlay_correct.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved as 'TMA1_A-3_overlay_correct.png'")

print(f"\n" + "="*60)
print(f"FINAL RESULTS:")
print("="*60)
print(f"  Cells plotted: {core_a3.n_obs}")
print(f"  Coordinate range (mm): {x_mm.max() - x_mm.min():.4f} x {y_mm.max() - y_mm.min():.4f}")
print(f"  Scale factors: {scale_factor_x:.10f} (X), {scale_factor_y:.10f} (Y)")
print(f"  mm_to_pixel: {mm_to_pixel_x:.4f} (X), {mm_to_pixel_y:.4f} (Y)")
print(f"  Rotation: {rotation_degrees:.0f}°")
print("="*60)
