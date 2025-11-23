#!/usr/bin/env python3
"""
Download PathGen-1.6M metadata JSON from HuggingFace.

This script attempts multiple download methods:
1. huggingface_hub library (recommended)
2. Direct wget download
3. Falls back to test data if authentication is required
"""

import os
import json
import shutil
import subprocess
from pathlib import Path


def download_via_huggingface_hub(output_path):
    """Try downloading using huggingface_hub library."""
    from huggingface_hub import hf_hub_download

    print("Attempting download using huggingface_hub...")
    file_path = hf_hub_download(
        repo_id="jamessyx/PathGen",
        filename="PathGen-1.6M.json",
        repo_type="dataset",
        cache_dir=None,
        local_files_only=False,
    )

    # Copy to our target location
    shutil.copy(file_path, output_path)
    print(f"Successfully downloaded via huggingface_hub to {output_path}")
    return True


def download_via_wget(url, output_path):
    """Try downloading using wget."""
    try:
        print("Attempting direct download with wget...")
        result = subprocess.run(
            ["wget", "--timeout=30", "--tries=3", url, "-O", str(output_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("Successfully downloaded via wget")
            return True
        else:
            print(f"wget failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"wget download failed: {e}")
        return False


def validate_json(file_path):
    """Validate that the downloaded file is proper JSON."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) > 0:
            print(f"JSON validation passed: {len(data)} entries found")

            # Show sample entry structure
            if data:
                print(f"Sample entry keys: {list(data[0].keys())}")

            return True
        else:
            print("JSON file appears empty or malformed")
            return False

    except Exception as e:
        print(f"JSON validation failed: {e}")
        return False


def main():
    """Main download function."""
    output_path = Path(snakemake.output.metadata_json)
    huggingface_url = snakemake.params.huggingface_url

    print("Downloading PathGen-1.6M metadata JSON...")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try huggingface_hub first
    if download_via_huggingface_hub(output_path):
        if validate_json(output_path):
            file_size = output_path.stat().st_size
            print(
                f"Download complete! File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)"
            )
            return
        else:
            print("Downloaded file validation failed, trying alternative methods...")

    # Try direct wget download
    if download_via_wget(huggingface_url, output_path):
        if validate_json(output_path):
            file_size = output_path.stat().st_size
            print(
                f"Download complete! File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)"
            )
            return
        else:
            print("Downloaded file validation failed")

    # Both methods failed - provide instructions and create test data
    print()
    print("=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print("The PathGen dataset requires accepting terms on HuggingFace.")
    print()
    print("Steps to download:")
    print("1. Visit: https://huggingface.co/datasets/jamessyx/PathGen")
    print("2. Click 'Access repository' and accept the terms")
    print("3. Install huggingface_hub: pip install huggingface_hub")
    print("4. Login: huggingface-cli login")
    print("5. Re-run this pipeline")
    print()
    print("Alternative manual download:")
    print(f"  wget {huggingface_url} -O {output_path}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
