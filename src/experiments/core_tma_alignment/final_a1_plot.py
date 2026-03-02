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
img_height, img_width = img.shape[:2]
print(f"Image shape: {img.shape} (H x W x C)")
print(f"Image dimensions: {img_width} x {img_height} pixels")

# Try different ways to find A-1 data
print("\n" + "="*60)
print("Searching for core A-1 in the dataset...")
print("="*60)

# Method 1: Check coreID column
a1_mask = adata.obs['coreID'] == 'A-1'
if a1_mask.sum() > 0:
    core_a1 = adata[a1_mask]
    print(f"✓ Found {core_a1.n_obs} cells with coreID='A-1'")
else:
    print(f"✗ No cells with coreID='A-1'")
    
    # Method 2: Check TMA_coreID column
    a1_mask = adata.obs['TMA_coreID'] == 'TMA1_A-1'
    if a1_mask.sum() > 0:
        core_a1 = adata[a1_mask]
        print(f"✓ Found {core_a1.n_obs} cells with TMA_coreID='TMA1_A-1'")
    else:
        print(f"✗ No cells with TMA_coreID='TMA1_A-1'")
        
        # Method 3: Check if there's a FOV that might correspond to A-1
        # Sometimes FOV 1 or first FOV corresponds to A-1
        print(f"\nCore A-1 not found in dataset!")
        print(f"Available cores in coreID: {sorted(adata.obs['coreID'].unique())[:10]}")
        print(f"Available cores in TMA_coreID: {sorted(adata.obs['TMA_coreID'].unique())[:10]}")
        print(f"\nThe h5ad file may only contain data for cores that passed QC.")
        print(f"The TIFF file 'TMA1_A-1.tiff' exists, but there's no matching data in the h5ad file.")
        exit()

# If we found data, proceed with plotting
print(f"\n" + "="*60)
print(f"Analyzing coordinates for core A-1 ({core_a1.n_obs} cells)")
print("="*60)

# Get coordinates
x_coords = core_a1.obs['x_FOV_px'].values
y_coords = core_a1.obs['y_FOV_px'].values

print(f"\nOriginal coordinate ranges (FOV pixels):")
print(f"  X: [{x_coords.min():.2f}, {x_coords.max():.2f}]")
print(f"  Y: [{y_coords.min():.2f}, {y_coords.max():.2f}]")
print(f"  X span: {x_coords.max() - x_coords.min():.2f} pixels")
print(f"  Y span: {y_coords.max() - y_coords.min():.2f} pixels")

# Calculate scaling factor to fit coordinates into image
x_min, x_max = x_coords.min(), x_coords.max()
y_min, y_max = y_coords.min(), y_coords.max()

x_range = x_max - x_min
y_range = y_max - y_min

scale_x = img_width / x_range if x_range > 0 else 1
scale_y = img_height / y_range if y_range > 0 else 1

print(f"\n" + "="*60)
print(f"Scaling calculation:")
print(f"  Image size: {img_width} x {img_height} pixels")
print(f"  Coordinate range: {x_range:.2f} x {y_range:.2f}")
print(f"  Scaling factor X: {scale_x:.6f}")
print(f"  Scaling factor Y: {scale_y:.6f}")
print("="*60)

# Scale coordinates to fit the image
x_scaled = (x_coords - x_min) * scale_x
y_scaled = (y_coords - y_min) * scale_y

print(f"\nScaled coordinate ranges (image pixels):")
print(f"  X: [{x_scaled.min():.2f}, {x_scaled.max():.2f}]")
print(f"  Y: [{y_scaled.min():.2f}, {y_scaled.max():.2f}]")

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.imshow(img)
ax.scatter(x_scaled, y_scaled, c='red', s=2, alpha=0.6, edgecolors='none')
ax.set_title(f'Core A-1: {core_a1.n_obs} cells overlaid on TMA1_A-1.tiff\n' + 
             f'Scaling: {scale_x:.4f}x (X), {scale_y:.4f}x (Y)', fontsize=14)
ax.set_xlabel('X (pixels)', fontsize=12)
ax.set_ylabel('Y (pixels)', fontsize=12)
ax.set_xlim(0, img_width)
ax.set_ylim(img_height, 0)  # Invert Y-axis to match image coordinates
plt.tight_layout()
plt.savefig('TMA1_A-1_overlay.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved as 'TMA1_A-1_overlay.png'")
print(f"\nFinal scaling factors:")
print(f"  X: {scale_x:.6f}")
print(f"  Y: {scale_y:.6f}")
