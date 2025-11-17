"""
Split lymphoma CosMx dataset into training and validation sets by cores.

Core purpose: Split the processed dataset by core IDs, with specified cores
going to validation and the rest to training.
"""

import anndata
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

# Get parameters from snakemake
validation_cores = snakemake.params.validation_cores
dataset = snakemake.wildcards.dataset

logging.info(f"Splitting {dataset} into training and validation sets")
logging.info(f"Validation cores: {validation_cores}")

# Load the processed dataset
logging.info(f"Loading dataset from {snakemake.input.adata}")
adata = anndata.read_h5ad(snakemake.input.adata)

logging.info(f"Loaded dataset with {adata.n_obs} observations and {adata.n_vars} variables")

# Check if core_id column exists
if "core_id" not in adata.obs.columns:
    logging.warning("No 'core_id' column found in dataset. Cannot split by cores.")
    logging.info("Creating dummy split based on random sampling...")
    
    # Fallback: random 80/20 split if no core information
    import numpy as np
    np.random.seed(42)
    n_obs = adata.n_obs
    validation_size = int(0.2 * n_obs)
    validation_indices = np.random.choice(n_obs, size=validation_size, replace=False)
    
    validation_mask = pd.Series(False, index=adata.obs_names)
    validation_mask.iloc[validation_indices] = True
    training_mask = ~validation_mask
    
    logging.info(f"Random split: {training_mask.sum()} training, {validation_mask.sum()} validation")

else:
    # Split by core IDs
    logging.info("Splitting by core IDs")
    
    # Get unique cores in the dataset
    available_cores = adata.obs['core_id'].dropna().unique()
    logging.info(f"Available cores in dataset: {sorted(available_cores)}")
    
    # Check which validation cores are actually present
    present_validation_cores = [core for core in validation_cores if core in available_cores]
    missing_validation_cores = [core for core in validation_cores if core not in available_cores]
    
    if missing_validation_cores:
        logging.warning(f"Validation cores not found in dataset: {missing_validation_cores}")
    if present_validation_cores:
        logging.info(f"Using validation cores: {present_validation_cores}")
    else:
        logging.error("No validation cores found in dataset!")
        raise ValueError("None of the specified validation cores are present in the dataset")
    
    # Create masks for training and validation
    validation_mask = adata.obs['core_id'].isin(present_validation_cores)
    training_mask = ~validation_mask
    
    # Handle cells with missing core_id (assign to training by default)
    missing_core_mask = adata.obs['core_id'].isna()
    if missing_core_mask.sum() > 0:
        logging.info(f"Found {missing_core_mask.sum()} cells with missing core_id, assigning to training")
        training_mask = training_mask | missing_core_mask
        validation_mask = validation_mask & ~missing_core_mask
    
    logging.info(f"Core-based split: {training_mask.sum()} training, {validation_mask.sum()} validation")
    
    # Log core distribution
    if validation_mask.sum() > 0:
        val_cores = adata.obs[validation_mask]['core_id'].value_counts()
        logging.info(f"Validation core counts:\n{val_cores}")
    
    if training_mask.sum() > 0:
        train_cores = adata.obs[training_mask]['core_id'].value_counts().head(10)
        logging.info(f"Training core counts (top 10):\n{train_cores}")

# Create training and validation datasets
logging.info("Creating training dataset...")
training_adata = adata[training_mask].copy()

logging.info("Creating validation dataset...")
validation_adata = adata[validation_mask].copy()

# Save the datasets
logging.info(f"Saving training dataset to {snakemake.output.training_adata}")
training_adata.write_h5ad(snakemake.output.training_adata)

logging.info(f"Saving validation dataset to {snakemake.output.validation_adata}")
validation_adata.write_h5ad(snakemake.output.validation_adata)

logging.info("Split complete!")
logging.info(f"Training: {training_adata.n_obs} observations")
logging.info(f"Validation: {validation_adata.n_obs} observations")
logging.info(f"Total: {training_adata.n_obs + validation_adata.n_obs} observations")