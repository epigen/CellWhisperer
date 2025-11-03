"""
Extract sample-level metadata (obs) from retrieval datasets
"""

import pandas as pd
import anndata as sc
from pathlib import Path


combined_obs = []


# Extract metadata from all input files
for file_path in sorted(snakemake.input.dataset_files):
    adata = sc.read_h5ad(file_path, backed="r")
    obs_df = adata.obs.copy()

    # Add sample_id from uns if available
    if "sample_id" in adata.uns:
        obs_df["sample_id"] = adata.uns["sample_id"]
    else:
        # Use filename as fallback
        obs_df["sample_id"] = Path(file_path).stem

    combined_obs.append(obs_df)

final_metadata = pd.concat(combined_obs, ignore_index=False)

# Add dataset name for reference
final_metadata["dataset"] = snakemake.wildcards.dataset

# Save to output
final_metadata.to_csv(snakemake.output.metadata, index=True)
