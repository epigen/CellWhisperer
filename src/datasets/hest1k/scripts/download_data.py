"""
Download HEST-1K dataset from HuggingFace.

This script downloads HEST-1K samples using the HEST library.
Processing is handled separately in process_data.py.
"""

from pathlib import Path
from huggingface_hub import login
from hest import iter_hest
import datasets


# Login to HuggingFace
if snakemake.params.huggingface_token:
    login(token=snakemake.params.huggingface_token)
else:
    print(
        "Warning: No HuggingFace token provided. Attempting to use cached credentials."
    )

# Create cache directory
cache_dir = Path(snakemake.params.hest_cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

# Get individual sample ID
sample_id = snakemake.params.sample_id
print(f"Downloading HEST data for sample: {sample_id}")

# Create pattern for specific sample ID
pattern = f"*{sample_id}[_.]**"

# Download the dataset using HEST's recommended approach
dataset = datasets.load_dataset(
    "MahmoodLab/hest",
    cache_dir=cache_dir,
    patterns=[pattern],
    trust_remote_code=True,
    # download_mode="force_redownload",
)

print(f"Sample {sample_id} downloaded successfully")

# Create flag file to indicate successful download
flag_file = Path(snakemake.output.download_flag)
flag_file.touch()
