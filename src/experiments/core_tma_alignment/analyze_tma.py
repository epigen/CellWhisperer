import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile

# Read the h5ad file
print("Loading h5ad file...")
adata = sc.read_h5ad('TMA1.h5ad')

print(f'Total cells in dataset: {adata.n_obs}')
print(f'\nAvailable obs columns: {list(adata.obs.columns)}')
print(f'\nAvailable obsm keys: {list(adata.obsm.keys())}')

# Check first few rows to understand the data structure
print(f'\nFirst few obs entries:')
print(adata.obs.head())

# Check unique values in fov column (if it exists)
if 'fov' in adata.obs.columns:
    print(f'\nUnique FOV values: {adata.obs["fov"].unique()[:10]}')
