#!/usr/bin/env python3
"""
Run SpotWhisperer inference on HEST benchmark patches.

This script creates a CustomInferenceEncoder wrapper for SpotWhisperer and runs
inference on HEST patches, saving embeddings in HEST's expected format.
"""
import pyarrow  # to prevent weird library loading issues
import os
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import h5py
from PIL import Image

from cellwhisperer.utils.model_io import load_cellwhisperer_model
from torch.utils.data import Dataset


class H5HESTDataset(Dataset):
    """Dataset to read ST + H&E from .h5
    Copied from hestcore.datasets
    """

    def __init__(self, h5_path, img_transform=None, chunk_size=1000):
        self.h5_path = h5_path
        self.img_transform = img_transform
        self.chunk_size = chunk_size
        with h5py.File(h5_path, "r") as f:
            self.n_chunks = int(np.ceil(len(f["barcode"]) / chunk_size))

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = (idx + 1) * self.chunk_size
        with h5py.File(self.h5_path, "r") as f:
            imgs = f["img"][start_idx:end_idx]
            barcodes = f["barcode"][start_idx:end_idx].flatten().tolist()
            coords = f["coords"][start_idx:end_idx]

        if self.img_transform:
            imgs = torch.stack(
                [self.img_transform(Image.fromarray(img)) for img in imgs]
            )

        return {"imgs": imgs, "barcodes": barcodes, "coords": coords}


def save_hdf5(
    output_fpath, asset_dict, attr_dict=None, mode="a", auto_chunk=True, chunk_size=None
):
    """
    Copied from HEST/src/hest/bench/utils/file_utils.py
    output_fpath: str, path to save h5 file
    asset_dict: dict, dictionary of key, val to save
    attr_dict: dict, dictionary of key: {k,v} to save as attributes for each key
    mode: str, mode to open h5 file
    auto_chunk: bool, whether to use auto chunking
    chunk_size: if auto_chunk is False, specify chunk size
    """
    with h5py.File(output_fpath, mode) as f:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if len(data_shape) == 1:
                val = np.expand_dims(val, axis=1)
                data_shape = val.shape

            if key not in f:  # if key does not exist, create dataset
                data_type = val.dtype
                if data_type == np.object_:
                    data_type = h5py.string_dtype(encoding="utf-8")
                if auto_chunk:
                    chunks = True  # let h5py decide chunk size
                else:
                    chunks = (chunk_size,) + data_shape[1:]
                try:
                    dset = f.create_dataset(
                        key,
                        shape=data_shape,
                        chunks=chunks,
                        maxshape=(None,) + data_shape[1:],
                        dtype=data_type,
                    )
                    ### Save attribute dictionary
                    if attr_dict is not None:
                        if key in attr_dict.keys():
                            for attr_key, attr_val in attr_dict[key].items():
                                dset.attrs[attr_key] = attr_val
                    dset[:] = val
                except:
                    print(f"Error encoding {key} of dtype {data_type} into hdf5")

            else:
                dset = f[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                assert dset.dtype == val.dtype
                dset[-data_shape[0] :] = val

        # if attr_dict is not None:
        #     for key, attr in attr_dict.items():
        #         if (key in asset_dict.keys()) and (len(asset_dict[key].attrs.keys())==0):
        #             for attr_key, attr_val in attr.items():
        #                 dset[key].attrs[attr_key] = attr_val

    return output_fpath


logger = logging.getLogger("run_spotwhisperer_inference")


class SpotWhispererInferenceEncoder:
    """
    Custom inference encoder wrapper for SpotWhisperer to interface with HEST benchmark.

    This class follows the HEST CustomInferenceEncoder pattern and wraps SpotWhisperer
    to make it compatible with HEST's benchmarking infrastructure.
    """

    def __init__(self, model_path, image_processor=None):
        # Load SpotWhisperer model
        (
            self.pl_model,
            self.text_processor,
            self.transcriptome_processor,
            self.image_processor,
        ) = load_cellwhisperer_model(model_path=model_path, eval=True)

        self.model = self.pl_model.model
        self.eval_transforms = self.image_processor.transform
        self.precision = torch.float32

        logger.info(f"SpotWhisperer model loaded: {type(self.model)}")
        logger.info(f"Image model: {type(self.model.image_model)}")

    def to(self, device):
        """Move model to specified device"""
        self.model = self.model.to(device)
        return self

    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()
        return self

    def __call__(self, patches):
        """
        Forward pass for patch inference.

        Args:
            patches: torch.Tensor of shape (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: Image embeddings
        """
        # SpotWhisperer expects multi-scale patches: (batch_size, n_scales, 3, 224, 224)
        batch_size = patches.shape[0]
        n_scales = len(self.model.image_model.config.scale_factors)

        # Replicate patches across all scales (simple approach)
        multi_scale_patches = patches.unsqueeze(1).expand(-1, n_scales, -1, -1, -1)

        # Get embeddings using SpotWhisperer's get_image_features method
        _, image_embeds = self.model.get_image_features(
            patches=multi_scale_patches, normalize_embeds=True
        )

        return image_embeds


def get_path(path):
    """Convert relative path to absolute path"""
    if path.startswith("./"):
        new_path = os.path.abspath(os.path.join(".", path))
    else:
        new_path = path
    return new_path


def post_collate_fn(batch):
    """
    Post collate function to clean up batch (from HEST benchmark)
    """
    if batch["imgs"].dim() == 5:
        assert batch["imgs"].size(0) == 1
        batch["imgs"] = batch["imgs"].squeeze(0)
    if batch["coords"].dim() == 3:
        assert batch["coords"].size(0) == 1
        batch["coords"] = batch["coords"].squeeze(0)
    return batch


def embed_tiles_spotwhisperer(
    dataloader,
    encoder,
    embedding_save_path: str,
    device: str,
):
    """
    Extract embeddings from tiles using SpotWhisperer encoder and save to H5 file.

    This function is adapted from HEST's embed_tiles() to work with SpotWhisperer.
    """
    encoder.eval()

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = post_collate_fn(batch)
        imgs = batch["imgs"].to(device).float()

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=encoder.precision):
            embeddings = encoder(imgs)

        if batch_idx == 0:
            mode = "w"
        else:
            mode = "a"

        asset_dict = {"embeddings": embeddings.cpu().numpy()}
        asset_dict.update(
            {key: np.array(val) for key, val in batch.items() if key != "imgs"}
        )

        save_hdf5(embedding_save_path, asset_dict=asset_dict, mode=mode)

    return embedding_save_path


def process_dataset_splits(dataset_bench_path, embedding_dir, encoder, device):
    """
    Process all splits for a given dataset using SpotWhisperer encoder.

    This function follows HEST's approach of processing train/test splits for each dataset.
    """
    splits_dir = dataset_bench_path / "splits"

    if not splits_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")

    # Find all splits
    split_files = list(splits_dir.glob("*.csv"))
    train_splits = [f for f in split_files if f.name.startswith("train_")]
    test_splits = [f for f in split_files if f.name.startswith("test_")]

    logger.info(
        f"Found {len(train_splits)} train splits and {len(test_splits)} test splits"
    )

    # Process each split
    all_splits = train_splits + test_splits
    processed_samples = set()  # To avoid duplicate processing

    for split_file in all_splits:
        logger.info(f"Processing split: {split_file.name}")

        split_df = pd.read_csv(split_file)

        for i in tqdm(range(len(split_df)), desc=f"Processing {split_file.name}"):
            sample_id = split_df.iloc[i]["sample_id"]

            # Raise if already processed (since train/test should not have same samples)
            if sample_id in processed_samples:
                logging.warning(
                    f"Sample {sample_id} already processed, skipping duplicate."
                )
                continue

            patches_path = split_df.iloc[i]["patches_path"]
            tile_h5_path = dataset_bench_path / patches_path

            embed_path = embedding_dir / f"{sample_id}.h5"

            # Skip if embeddings already exist
            if embed_path.exists():
                logger.info(f"Skipping {sample_id} - embeddings already exist")
                processed_samples.add(sample_id)
                continue

            # Create dataset and dataloader for this sample
            tile_dataset = H5HESTDataset(
                str(tile_h5_path),
                chunk_size=snakemake.params.batch_size,
                img_transform=encoder.eval_transforms,
            )

            tile_dataloader = torch.utils.data.DataLoader(
                tile_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0 if snakemake.threads == 1 else snakemake.threads,
            )

            # Extract embeddings
            embed_tiles_spotwhisperer(tile_dataloader, encoder, str(embed_path), device)
            processed_samples.add(sample_id)

            logger.info(f"Completed embeddings for {sample_id}")

    logger.info(f"Completed processing {len(processed_samples)} samples")


# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Create SpotWhisperer encoder wrapper
logger.info("Loading SpotWhisperer model...")
encoder = SpotWhispererInferenceEncoder(
    model_path=snakemake.input.model,
    image_processor=None,  # Will be loaded inside the class
)
encoder.to(device)

# Create output directory
Path(snakemake.output.embeddings_dir).mkdir(parents=True, exist_ok=True)

# Extract dataset name from wildcards and use input/output paths directly
dataset_bench_path = Path(snakemake.input.dataset_dir)

embeddings_dir = Path(snakemake.output.embeddings_dir)

logger.info(f"Dataset bench path: {dataset_bench_path}")
logger.info(f"Output embeddings dir: {snakemake.output.embeddings_dir}")

process_dataset_splits(
    dataset_bench_path, Path(snakemake.output.embeddings_dir), encoder, device
)

logger.info("SpotWhisperer inference completed successfully")
