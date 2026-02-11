#!/usr/bin/env python3
"""
Test script to verify both lymphoma_cosmx resolution variants can be generated.

This script tests:
1. lymphoma_cosmx_small (Visium-sized spots, 55px diameter)
2. lymphoma_cosmx_small_detailed (high-resolution spots, 20px diameter)
"""

import subprocess
import sys
from pathlib import Path
import anndata
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(cmd, cwd=None):
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.error(f"Command failed: {cmd}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"Exception running command {cmd}: {e}")
        return False


def test_dataset_generation():
    """Test generation of both dataset variants."""

    # Get project root
    try:
        project_root = (
            subprocess.check_output("git rev-parse --show-toplevel", shell=True)
            .decode("utf-8")
            .strip()
        )
        project_root = Path(project_root)
    except:
        logger.error("Could not find git repository root")
        return False

    # Change to lymphoma_cosmx_small directory
    dataset_dir = project_root / "src/datasets/lymphoma_cosmx_small"

    logger.info("Creating test data...")
    if not run_command("pixi run python create_test_data.py", cwd=dataset_dir):
        logger.error("Failed to create test data")
        return False

    logger.info("Test data created successfully")

    # Test both dataset variants
    datasets_to_test = ["lymphoma_cosmx_small", "lymphoma_cosmx_small_detailed"]

    for dataset_name in datasets_to_test:
        logger.info(f"Testing dataset generation for: {dataset_name}")

        # Run snakemake for this specific dataset
        cmd = f"pixi run snakemake -R process_data -j1 {project_root}/results/{dataset_name}/data.h5ad"
        if not run_command(cmd, cwd=dataset_dir):
            logger.error(f"Failed to generate {dataset_name}")
            return False

        # Verify the output file exists and is valid
        output_file = project_root / f"results/{dataset_name}/data.h5ad"
        if not output_file.exists():
            logger.error(f"Output file not found: {output_file}")
            return False

        # Load and validate the AnnData object
        try:
            adata = anndata.read_h5ad(output_file)
            logger.info(f"{dataset_name} - Shape: {adata.shape}")
            logger.info(
                f"{dataset_name} - Spatial coords shape: {adata.obsm['spatial'].shape}"
            )
            logger.info(
                f"{dataset_name} - Pixel coordinates range: x=[{adata.obs['x_pixel'].min()}-{adata.obs['x_pixel'].max()}], y=[{adata.obs['y_pixel'].min()}-{adata.obs['y_pixel'].max()}]"
            )
            logger.info(
                f"{dataset_name} - Cell counts per tile: mean={adata.obs['n_cells'].mean():.1f}, max={adata.obs['n_cells'].max()}"
            )

            # Verify required columns exist
            required_obs_cols = ["x_array", "y_array", "x_pixel", "y_pixel", "n_cells"]
            for col in required_obs_cols:
                if col not in adata.obs.columns:
                    logger.error(f"{dataset_name} - Missing required column: {col}")
                    return False

            # Verify spatial coordinates exist
            if "spatial" not in adata.obsm:
                logger.error(f"{dataset_name} - Missing spatial coordinates in obsm")
                return False

            logger.info(f"{dataset_name} - Validation passed ✓")

        except Exception as e:
            logger.error(f"Failed to load/validate {dataset_name}: {e}")
            return False

    return True


def compare_resolutions():
    """Compare the two resolution variants."""
    try:
        project_root = (
            subprocess.check_output("git rev-parse --show-toplevel", shell=True)
            .decode("utf-8")
            .strip()
        )
        project_root = Path(project_root)

        # Load both datasets
        small_file = project_root / "results/lymphoma_cosmx_small/data.h5ad"
        detailed_file = project_root / "results/lymphoma_cosmx_small_detailed/data.h5ad"

        if not (small_file.exists() and detailed_file.exists()):
            logger.error("Both dataset files must exist for comparison")
            return False

        adata_small = anndata.read_h5ad(small_file)
        adata_detailed = anndata.read_h5ad(detailed_file)

        logger.info("\n=== RESOLUTION COMPARISON ===")
        logger.info(f"lymphoma_cosmx_small:    {adata_small.shape[0]} tiles")
        logger.info(f"lymphoma_cosmx_small_detailed: {adata_detailed.shape[0]} tiles")
        logger.info(
            f"Ratio (detailed/small):  {adata_detailed.shape[0] / adata_small.shape[0]:.2f}x more tiles"
        )

        # Compare cell density
        small_density = adata_small.obs["n_cells"].mean()
        detailed_density = adata_detailed.obs["n_cells"].mean()
        logger.info(
            f"Average cells per tile - small: {small_density:.1f}, detailed: {detailed_density:.1f}"
        )

        # The detailed version should have more tiles but fewer cells per tile on average
        if adata_detailed.shape[0] <= adata_small.shape[0]:
            logger.warning(
                "Expected detailed version to have more tiles than small version"
            )

        if detailed_density >= small_density:
            logger.warning(
                "Expected detailed version to have fewer cells per tile on average"
            )

        return True

    except Exception as e:
        logger.error(f"Failed to compare resolutions: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting test of both lymphoma_cosmx resolution variants...")

    # Test dataset generation
    if not test_dataset_generation():
        logger.error("Dataset generation test failed")
        sys.exit(1)

    # Compare the results
    if not compare_resolutions():
        logger.error("Resolution comparison failed")
        sys.exit(1)

    logger.info("All tests passed! Both resolution variants generated successfully.")
