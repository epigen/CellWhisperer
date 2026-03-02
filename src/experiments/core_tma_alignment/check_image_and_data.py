import scanpy as sc
import numpy as np
from PIL import Image
import tifffile

# Load the TIFF image
print("Loading TIFF image...")
img = tifffile.imread('TMA1_A-1.tiff')
print(f"Image shape: {img.shape}")
print(f"Image dtype: {img.dtype}")

# Read the h5ad file
print("\nLoading h5ad file...")
adata = sc.read_h5ad('TMA1.h5ad')

# Check if there are any cells with coordinates that might match this image
# Let's look at the FOV values more carefully
print(f"\nChecking FOV column...")
print(f"Unique FOV values: {sorted(adata.obs['fov'].unique())}")

# Check the tmaBatch_fov column
if 'tmaBatch_fov' in adata.obs.columns:
    print(f"\nUnique tmaBatch_fov values (first 20): {sorted(adata.obs['tmaBatch_fov'].unique())[:20]}")

# Let's also check if there's any reference to 'A-1' or 'A_1' in any column
print("\nSearching for A-1 or A_1 patterns in all string columns...")
for col in adata.obs.columns:
    if adata.obs[col].dtype == 'object':
        unique_vals = adata.obs[col].unique()
        matching = [v for v in unique_vals if isinstance(v, str) and ('A-1' in v or 'A_1' in v)]
        if matching:
            print(f"  {col}: {matching[:5]}")
