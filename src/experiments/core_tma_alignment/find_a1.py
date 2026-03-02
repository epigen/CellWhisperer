import scanpy as sc
import numpy as np

# Read the h5ad file
adata = sc.read_h5ad('TMA1.h5ad')

# Check all unique TMA_coreID values
tma_cores = sorted(adata.obs['TMA_coreID'].unique())
print(f"All TMA_coreID values containing 'A-1':")
a1_cores = [c for c in tma_cores if 'A-1' in c]
print(a1_cores)

print(f"\nAll cores starting with TMA1_A:")
tma1_a = [c for c in tma_cores if c.startswith('TMA1_A')]
print(tma1_a[:15])

# Try filtering with TMA1_A-1
if 'TMA1_A-1' in tma_cores:
    core_a1 = adata[adata.obs['TMA_coreID'] == 'TMA1_A-1']
    print(f"\nCells in TMA1_A-1: {core_a1.n_obs}")
else:
    print(f"\nTMA1_A-1 not found in dataset")
    print(f"Perhaps using the first available core for testing: {tma_cores[0]}")
