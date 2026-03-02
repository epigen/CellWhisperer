import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import tifffile

# Read the h5ad file
print("Loading h5ad file...")
adata = sc.read_h5ad('TMA1.h5ad')

# Load the TIFF image
print("Loading TIFF image...")
img = tifffile.imread('TMA1_A-1.tiff')
print(f"Image shape: {img.shape}")

# Since A-1 doesn't exist, let's use the first available FOV (FOV 3)
# to demonstrate the approach
first_fov = sorted(adata.obs['fov'].unique())[0]
print(f"\nUsing FOV {first_fov} as example (since A-1 doesn't exist in dataset)")

# Filter for first FOV
fov_data = adata[adata.obs['fov'] == first_fov]
print(f"Cells in FOV {first_fov}: {fov_data.n_obs}")

# Get coordinates - try different coordinate systems
if 'x_FOV_px' in fov_data.obs.columns and 'y_FOV_px' in fov_data.obs.columns:
    x_coords = fov_data.obs['x_FOV_px'].values
    y_coords = fov_data.obs['y_FOV_px'].values
    coord_type = "FOV pixels"
elif 'x_slide_mm_transform' in fov_data.obs.columns and 'y_slide_mm_transform' in fov_data.obs.columns:
    x_coords = fov_data.obs['x_slide_mm_transform'].values
    y_coords = fov_data.obs['y_slide_mm_transform'].values
    coord_type = "slide mm (transformed)"
else:
    print("No suitable coordinates found")
    exit()

print(f"\nCoordinate system: {coord_type}")
print(f"X range: [{x_coords.min():.4f}, {x_coords.max():.4f}]")
print(f"Y range: [{y_coords.min():.4f}, {y_coords.max():.4f}]")
print(f"X span: {x_coords.max() - x_coords.min():.4f}")
print(f"Y span: {y_coords.max() - y_coords.min():.4f}")

# Check if coordinates need scaling to match image size
img_height, img_width = img.shape[:2]
print(f"\nImage dimensions: {img_width} x {img_height} pixels")

# Calculate scaling factor
x_range = x_coords.max() - x_coords.min()
y_range = y_coords.max() - y_coords.min()

if x_range < img_width and y_range < img_height:
    print("\nCoordinates appear to be in pixel space already (or smaller than image)")
    scale_x = img_width / x_range if x_range > 0 else 1
    scale_y = img_height / y_range if y_range > 0 else 1
    print(f"Suggested scaling: X={scale_x:.2f}, Y={scale_y:.2f}")
else:
    print("\nCoordinates may need to be scaled down or are in different units")
    scale_x = img_width / x_range
    scale_y = img_height / y_range
    print(f"Suggested scaling: X={scale_x:.6f}, Y={scale_y:.6f}")

# Scale coordinates to image space
x_scaled = (x_coords - x_coords.min()) * scale_x
y_scaled = (y_coords - y_coords.min()) * scale_y

print(f"\nScaled coordinates:")
print(f"X range: [{x_scaled.min():.2f}, {x_scaled.max():.2f}] pixels")
print(f"Y range: [{y_scaled.min():.2f}, {y_scaled.max():.2f}] pixels")

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(img)
ax.scatter(x_scaled, y_scaled, c='red', s=1, alpha=0.5)
ax.set_title(f'FOV {first_fov} cells overlaid on TMA1_A-1.tiff\nScaling factor: {scale_x:.4f}x, {scale_y:.4f}y')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
plt.tight_layout()
plt.savefig('overlay_plot.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved as 'overlay_plot.png'")
print(f"Scaling factor: X={scale_x:.6f}, Y={scale_y:.6f}")
