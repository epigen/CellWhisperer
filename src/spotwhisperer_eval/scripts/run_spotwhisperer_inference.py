#!/usr/bin/env python3
"""
Run SpotWhisperer inference on HEST benchmark patches.

This script creates a CustomInferenceEncoder wrapper for SpotWhisperer and runs
inference on HEST patches, saving embeddings in HEST's expected format.

NOTE: This file could be drastically refactored to use spotwhisperer-style code. Or better, the hest benchmark could be transformed to a separate dataset so that we could run `cellwhisperer test` directly
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
import scanpy as sc


def normalize_adata(adata: sc.AnnData, smooth=False) -> sc.AnnData:
    """
    Normalize each spot by total gene counts + Logarithmize each spot
    Copied from modules/HEST/src/hest/bench/st_dataset.py
    """
    filtered_adata = adata.copy()
    filtered_adata.X = filtered_adata.X.astype(np.float64)

    if smooth:
        adata_df = adata.to_df()
        for index, df_row in adata.obs.iterrows():
            row = int(df_row["array_row"])
            col = int(df_row["array_col"])
            neighbors_index = adata.obs[
                (
                    (adata.obs["array_row"] >= row - 1)
                    & (adata.obs["array_row"] <= row + 1)
                )
                & (
                    (adata.obs["array_col"] >= col - 1)
                    & (adata.obs["array_col"] <= col + 1)
                )
            ].index
            neighbors = adata_df.loc[neighbors_index]
            nb_neighbors = len(neighbors)

            avg = neighbors.sum() / nb_neighbors
            filtered_adata[index] = avg

    # Logarithm of the expression
    sc.pp.log1p(filtered_adata)
    return filtered_adata


def load_adata(expr_path, genes=None, barcodes=None, normalize=False):
    """
    Load expression data from .h5ad file
    Copied from modules/HEST/src/hest/bench/st_dataset.py
    """
    adata = sc.read_h5ad(expr_path)
    if barcodes is not None:
        adata = adata[barcodes]
    if genes is not None:
        adata = adata[:, genes]
    if normalize:
        adata = normalize_adata(adata)
    return adata.to_df()


class H5HESTDataset(Dataset):
    """Dataset to read ST + H&E from .h5
    Extended from hestcore.datasets to also load transcriptomics data
    """

    def __init__(
        self,
        h5_path,
        expr_path=None,
        genes=None,
        img_transform=None,
        chunk_size=1000,
        normalize_expr=True,
    ):
        self.h5_path = h5_path
        self.expr_path = expr_path
        self.genes = genes
        self.img_transform = img_transform
        self.chunk_size = chunk_size
        self.normalize_expr = normalize_expr

        with h5py.File(h5_path, "r") as f:
            self.n_chunks = int(np.ceil(len(f["barcode"]) / chunk_size))
            # Store all barcodes for transcriptomics loading
            self.all_barcodes = f["barcode"][:].flatten().astype(str).tolist()

        # Load expression data once if provided
        self.expression_data = None
        if expr_path is not None and genes is not None:
            logger.info(f"Loading expression data from {expr_path}")
            adata_df = load_adata(
                expr_path,
                genes=genes,
                barcodes=self.all_barcodes,
                normalize=normalize_expr,
            )
            self.expression_data = adata_df.values  # Convert to numpy array
            logger.info(f"Loaded expression data shape: {self.expression_data.shape}")

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

        batch_data = {"imgs": imgs, "barcodes": barcodes, "coords": coords}

        # Add expression data if available
        if self.expression_data is not None:
            batch_data["expression"] = self.expression_data[start_idx:end_idx]

        return batch_data


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

    def __call__(self, patches, expression_data=None):
        """
        Forward pass for patch and expression inference.

        Args:
            patches: torch.Tensor of shape (batch_size, 3, 224, 224)
            expression_data: numpy.ndarray of shape (batch_size, n_genes) or None

        Returns:
            dict: Dictionary with 'image_embeds' and optionally 'transcriptome_embeds'
        """
        results = {}

        # Process image patches
        # SpotWhisperer expects multi-scale patches: (batch_size, n_scales, 3, 224, 224)
        batch_size = patches.shape[0]
        # Build views dict for UNIModel (context and cell)
        views = {
            "context": patches,                    # (B,3,224,224)
            "cell": torch.nn.functional.interpolate(
                patches, size=(56, 56), mode="bilinear", align_corners=False
            ),
        }

        # Get image embeddings using SpotWhisperer's get_image_features method
        _, image_embeds = self.model.get_image_features(
            patches_ctx=patches,
            patches_cell=torch.nn.functional.interpolate(
                patches, size=(56, 56), mode="bilinear", align_corners=False
            ),
            normalize_embeds=True,
        )

        results["image_embeds"] = image_embeds

        # Process transcriptome data if provided
        if expression_data is not None:
            # Convert expression data to the format expected by SpotWhisperer's transcriptome processor
            expression_tensor = torch.tensor(
                expression_data, dtype=torch.float32, device=patches.device
            )

            # Process through transcriptome processor to get the input format for the model
            transcriptome_inputs = self.transcriptome_processor(
                expression_tensor, return_tensors="pt"
            )

            # Move to correct device
            for key in transcriptome_inputs:
                if isinstance(transcriptome_inputs[key], torch.Tensor):
                    transcriptome_inputs[key] = transcriptome_inputs[key].to(
                        patches.device
                    )

            # Get transcriptome embeddings using SpotWhisperer's get_transcriptome_features method
            _, transcriptome_embeds = self.model.get_transcriptome_features(
                **transcriptome_inputs, normalize_embeds=True
            )
            results["transcriptome_embeds"] = transcriptome_embeds

        return results


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
    Extract image and transcriptome embeddings from tiles using SpotWhisperer encoder and save to H5 file.

    This function is adapted from HEST's embed_tiles() to work with SpotWhisperer.
    """
    encoder.eval()

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = post_collate_fn(batch)
        imgs = batch["imgs"].to(device).float()

        # Get expression data if available
        expression_data = batch.get("expression", None)

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=encoder.precision):
            embedding_results = encoder(imgs, expression_data)

        if batch_idx == 0:
            mode = "w"
        else:
            mode = "a"

        # Save both image and transcriptome embeddings
        asset_dict = {
            "image_embeddings": embedding_results["image_embeds"].cpu().numpy()
        }

        if "transcriptome_embeds" in embedding_results:
            asset_dict["transcriptome_embeddings"] = (
                embedding_results["transcriptome_embeds"].cpu().numpy()
            )

        # Also save the legacy "embeddings" key for backward compatibility (image embeddings)
        asset_dict["embeddings"] = embedding_results["image_embeds"].cpu().numpy()

        # Add other batch data (excluding imgs and expression which are large)
        asset_dict.update(
            {
                key: np.array(val)
                for key, val in batch.items()
                if key not in ["imgs", "expression"]
            }
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
            expr_path = split_df.iloc[i]["expr_path"]
            tile_h5_path = dataset_bench_path / patches_path
            expr_full_path = dataset_bench_path / expr_path

            embed_path = embedding_dir / f"{sample_id}.h5"

            # Skip if embeddings already exist
            if embed_path.exists():
                logger.info(f"Skipping {sample_id} - embeddings already exist")
                processed_samples.add(sample_id)
                continue

            # Load genes list (assuming it's available in the dataset directory)
            genes_file = dataset_bench_path / "var_50genes.json"
            if genes_file.exists():
                import json

                with open(genes_file, "r") as f:
                    genes = json.load(f)["genes"]
                logger.info(f"Using {len(genes)} genes for transcriptome embeddings")
            else:
                genes = None
                logger.warning(
                    "No genes file found - will only generate image embeddings"
                )

            # Create dataset and dataloader for this sample
            tile_dataset = H5HESTDataset(
                str(tile_h5_path),
                expr_path=str(expr_full_path) if genes is not None else None,
                genes=genes,
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
