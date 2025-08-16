"""
STHELAR Dataset Downloads Script

This script downloads the STHELAR dataset from S-BIAD2146 archive.
Downloads both H&E patch images and zarr files with gene expression data.
"""

import os
import requests
import zipfile
from pathlib import Path
from urllib.parse import urljoin
import pandas as pd

# Base URLs for STHELAR dataset
BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/146/S-BIAD2146/"
DATA_20X_URL = urljoin(BASE_URL, "Files/STHELAR/data_20x/data/")
ZARR_URL = urljoin(BASE_URL, "Files/STHELAR/sdata_slides/")

def download_file(url, filepath, chunk_size=8192):
    """Download a file from URL with progress tracking."""
    print(f"Downloading {url} to {filepath}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    print()  # New line after progress

def get_slide_list():
    """Extract slide names from zarr file listing."""
    zarr_files = [
        "sdata_bone_marrow_s0.zarr.zip",
        "sdata_bone_marrow_s1.zarr.zip", 
        "sdata_bone_s0.zarr.zip",
        "sdata_brain_s0.zarr.zip",
        "sdata_breast_s0.zarr.zip",
        "sdata_breast_s1.zarr.zip",
        "sdata_breast_s3.zarr.zip",
        "sdata_breast_s6.zarr.zip",
        "sdata_cervix_s0.zarr.zip",
        "sdata_colon_s1.zarr.zip",
        "sdata_colon_s2.zarr.zip",
        "sdata_heart_s0.zarr.zip",
        "sdata_kidney_s0.zarr.zip",
        "sdata_kidney_s1.zarr.zip",
        "sdata_liver_s0.zarr.zip",
        "sdata_liver_s1.zarr.zip",
        "sdata_lung_s1.zarr.zip",
        "sdata_lung_s3.zarr.zip",
        "sdata_lymph_node_s0.zarr.zip",
        "sdata_ovary_s0.zarr.zip",
        "sdata_ovary_s1.zarr.zip",
        "sdata_pancreatic_s0.zarr.zip",
        "sdata_pancreatic_s1.zarr.zip",
        "sdata_pancreatic_s2.zarr.zip",
        "sdata_prostate_s0.zarr.zip",
        "sdata_skin_s1.zarr.zip",
        "sdata_skin_s2.zarr.zip",
        "sdata_skin_s3.zarr.zip",
        "sdata_skin_s4.zarr.zip",
        "sdata_tonsil_s0.zarr.zip",
        "sdata_tonsil_s1.zarr.zip"
    ]
    
    # Extract slide IDs from zarr filenames
    slides = []
    for zarr_file in zarr_files:
        # Extract slide ID: sdata_tissue_sX.zarr.zip -> tissue_sX
        slide_id = zarr_file.replace("sdata_", "").replace(".zarr.zip", "")
        slides.append(slide_id)
    
    return slides

def main():
    # Get parameters from snakemake
    slide_id = snakemake.params.slide_id
    cache_dir = Path(snakemake.params.sthelar_cache_dir)
    
    print(f"Downloading STHELAR data for slide: {slide_id}")
    
    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download images.zip (shared across all slides) - only once
    images_zip_path = cache_dir / "images.zip"
    if not images_zip_path.exists():
        images_url = urljoin(DATA_20X_URL, "images.zip")
        download_file(images_url, images_zip_path)
    else:
        print(f"Images.zip already exists at {images_zip_path}")
    
    # Download the zarr file for this specific slide
    zarr_filename = f"sdata_{slide_id}.zarr.zip"
    zarr_path = cache_dir / zarr_filename
    
    if not zarr_path.exists():
        zarr_url = urljoin(ZARR_URL, zarr_filename)
        download_file(zarr_url, zarr_path)
    else:
        print(f"Zarr file already exists at {zarr_path}")
    
    # Create flag file to indicate successful download
    flag_file = Path(snakemake.output.download_flag)
    flag_file.touch()
    
    print(f"Download completed for slide {slide_id}")

if __name__ == "__main__":
    main()