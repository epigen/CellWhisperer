#!/usr/bin/env python3
"""
Prepare HEST benchmark data.

This script downloads and prepares the HEST benchmark datasets with 
pre-extracted patches and split files, making them ready for SpotWhisperer inference.
"""

import os
import yaml
from pathlib import Path
from huggingface_hub import snapshot_download
from loguru import logger


def get_path(path):
    """Convert relative path to absolute path"""
    if path.startswith('./'):
        # Get the path relative to the current working directory
        new_path = os.path.abspath(os.path.join('.', path))
    else:
        new_path = path
    return new_path


def main():
    # Load configuration
    with open(snakemake.input.config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract dataset name from wildcards and use the output directory directly
    dataset_name = snakemake.wildcards.dataset
    dataset_path = Path(snakemake.output.dataset_dir)
    bench_data_root = str(dataset_path.parent)
    
    logger.info(f'Preparing HEST benchmark data for dataset: {dataset_name}')
    logger.info(f'Benchmark data will be saved to: {bench_data_root}')
    logger.info(f'Dataset will be available at: {dataset_path}')
    
    # Create directories
    Path(bench_data_root).mkdir(parents=True, exist_ok=True)
    
    try:
        # Download benchmark data (excluding model weights)
        logger.info('Downloading HEST benchmark datasets...')
        snapshot_download(
            repo_id="MahmoodLab/hest-bench", 
            repo_type='dataset', 
            local_dir=bench_data_root, 
            ignore_patterns=['fm_v1/*']
        )
        logger.info('Benchmark data downloaded successfully')
        
        # Verify that the required dataset is present
        logger.info(f'Verifying dataset: {dataset_name}')
        
        # Use the dataset_path directly (already defined from output)
        if not dataset_path.exists():
            logger.error(f'Dataset {dataset_name} not found in downloaded data')
            raise FileNotFoundError(f'Could not find required dataset: {dataset_name}')
        
        # Check for splits directory
        splits_path = dataset_path / 'splits'
        if not splits_path.exists():
            logger.error(f'No splits directory found for {dataset_name}')
            raise FileNotFoundError(f'No splits directory found for dataset: {dataset_name}')
        
        split_files = list(splits_path.glob('*.csv'))
        logger.info(f'Found {len(split_files)} split files for {dataset_name}')
        
        if not split_files:
            logger.error(f'No split files found for {dataset_name}')
            raise FileNotFoundError(f'No split files found for dataset: {dataset_name}')
        
        logger.info(f'Dataset {dataset_name} is ready')
        
    except Exception as e:
        logger.error(f'Error preparing HEST data for {dataset_name}: {str(e)}')
        raise
    
    # Ensure output directory exists (snakemake should create it, but just in case)
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"HEST data preparation completed successfully for dataset: {dataset_name}")


if __name__ == "__main__":
    main()