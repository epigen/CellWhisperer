"""
Randomly subsample a multi-file dataset by selecting 1/nth of the h5ad files.
Works by creating symlinks to selected files, preserving the directory structure.
"""
import glob
import random
import os
import numpy as np
from pathlib import Path

# Get parameters from snakemake
input_dir = Path(snakemake.input.h5ads_dir)
output_dir = Path(snakemake.output.subsampled_h5ads)
n = snakemake.params.n
seed = snakemake.params.seed

# Set random seed for reproducibility
random.seed(seed)
np.random.seed(seed)

# Find all h5ad files in the input directory
h5ad_files = list(input_dir.glob("*.h5ad"))

if not h5ad_files:
    raise ValueError(f"No h5ad files found in {input_dir}")

# Determine the number of files to keep (1/nth)
total_files = len(h5ad_files)
files_to_keep = max(1, total_files // n)

# Randomly sample file indices
selected_files = random.sample(h5ad_files, files_to_keep)

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Create symlinks to selected files
for file_path in selected_files:
    target_path = output_dir / file_path.name
    
    # Remove existing symlink if it exists
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    
    # Create relative symlink (more portable than absolute paths)
    relative_source = os.path.relpath(file_path, target_path.parent)
    os.symlink(relative_source, target_path)

print(f"Subsampled multi-file dataset from {total_files} to {len(selected_files)} files (1/{n}th)")
print(f"Created symlinks in {output_dir}")
print(f"Selected files: {[f.name for f in selected_files]}")