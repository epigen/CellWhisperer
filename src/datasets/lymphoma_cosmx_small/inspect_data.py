#!/usr/bin/env python3
"""
Script to inspect the actual h5ad file and see what columns are available.
"""

import anndata
from pathlib import Path

# Check both the test data and the real data
test_file = Path("resources/lymphoma_cosmx_small/read_count_table.h5ad")
real_file = Path("resources/lymphoma_cosmx_small/1-TON.h5ad")

for file_path in [test_file, real_file]:
    if file_path.exists():
        print(f"\n=== Inspecting {file_path} ===")
        try:
            adata = anndata.read_h5ad(file_path)
            print(f"Shape: {adata.shape}")
            print(f"obs columns: {list(adata.obs.columns)}")
            print(f"obsm keys: {list(adata.obsm.keys())}")
            print(f"uns keys: {list(adata.uns.keys())}")

            # Look for coordinate-related columns
            coord_cols = [
                col
                for col in adata.obs.columns
                if any(
                    coord in col.lower()
                    for coord in ["x", "y", "coord", "spatial", "mm", "pixel"]
                )
            ]
            if coord_cols:
                print(f"Coordinate-related columns: {coord_cols}")
                for col in coord_cols:
                    print(
                        f"  {col}: {adata.obs[col].dtype}, range: [{adata.obs[col].min():.3f}, {adata.obs[col].max():.3f}]"
                    )

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    else:
        print(f"\n{file_path} does not exist")
