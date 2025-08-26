#!/usr/bin/env python3
"""
Run HEST evaluation using pre-computed SpotWhisperer embeddings.

This script uses HEST's existing regression pipeline to evaluate SpotWhisperer embeddings
on the benchmark spatial transcriptomics prediction tasks.
"""

import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hest.bench.trainer import train_test_reg
from hest.bench.st_dataset import load_adata
from hest.bench.utils.file_utils import read_assets_from_h5, save_pkl
from hest.bench.utils.utils import merge_dict


def get_path(path):
    """Convert relative path to absolute path"""
    if path.startswith("./"):
        new_path = os.path.abspath(os.path.join(".", path))
    else:
        new_path = path
    return new_path


def merge_fold_results(results_arr):
    """
    Merge results from multiple folds (adapted from HEST's merge_fold_results)
    """
    aggr_dict = {}
    for result_dict in results_arr:
        for item in result_dict["pearson_corrs"]:
            gene_name = item["name"]
            correlation = item["pearson_corr"]
            aggr_dict[gene_name] = aggr_dict.get(gene_name, []) + [correlation]

    aggr_results = []
    all_corrs = []
    for key, value in aggr_dict.items():
        aggr_results.append(
            {
                "name": key,
                "pearson_corrs": value,
                "mean": np.mean(value),
                "std": np.std(value),
            }
        )
        all_corrs += value

    mean_per_split = [d["pearson_mean"] for d in results_arr]

    return {
        "pearson_corrs": aggr_results,
        "pearson_mean": np.mean(mean_per_split),
        "pearson_std": np.std(mean_per_split),
        "mean_per_split": mean_per_split,
    }


def evaluate_single_split(
    train_split_file, test_split_file, dataset_bench_path, embeddings_dir, results_dir
):
    """
    Evaluate a single train/test split using SpotWhisperer embeddings.

    This function is adapted from HEST's predict_single_split but focuses only on
    the evaluation part, using pre-computed embeddings from SpotWhisperer.
    """
    # Read split files
    train_df = pd.read_csv(train_split_file)
    test_df = pd.read_csv(test_split_file)

    logger.info(f"Loading embeddings from: {embeddings_dir}")

    with open(Path(snakemake.input.dataset_dir) / "var_50genes.json", "r") as f:
        genes = json.load(f)["genes"]

    logger.info(f"Using {len(genes)} genes for evaluation")

    # Load and gather all data for train and test splits
    all_split_assets = {}

    for split_key, split_df in zip(["train", "test"], [train_df, test_df]):
        split_assets = {}
        logger.info(f"Loading {split_key} split data...")

        for i in tqdm(range(len(split_df)), desc=f"Loading {split_key} samples"):
            sample_id = split_df.iloc[i]["sample_id"]
            embed_path = embeddings_dir / f"{sample_id}.h5"
            expr_path = dataset_bench_path / split_df.iloc[i]["expr_path"]

            # # Check if embedding file exists
            # if not embed_path.exists():
            #     logger.warning(f"Embedding file not found: {embed_path}")
            #     continue

            # # Check if expression file exists
            # if not expr_path.exists():
            #     logger.warning(f"Expression file not found: {expr_path}")
            #     continue

            # Load embeddings and metadata
            assets, _ = read_assets_from_h5(str(embed_path))
            barcodes = assets["barcodes"].flatten().astype(str).tolist()

            # Load expression data
            adata = load_adata(
                str(expr_path),
                genes=genes,
                barcodes=barcodes,
                normalize=snakemake.params.normalize,
            )
            assets["adata"] = adata.values

            # Merge assets
            split_assets = merge_dict(split_assets, assets)

        # Concatenate all data for this split
        if split_assets:
            for key, val in split_assets.items():
                split_assets[key] = np.concatenate(val, axis=0)

            all_split_assets[split_key] = split_assets
            logger.info(
                f"Loaded {split_key} split with {len(split_assets['embeddings'])} samples: {split_assets['embeddings'].shape}"
            )
        else:
            raise ValueError(f"No data loaded for {split_key} split")

    # Extract training and testing data
    X_train, y_train = (
        all_split_assets["train"]["embeddings"],
        all_split_assets["train"]["adata"],
    )
    X_test, y_test = (
        all_split_assets["test"]["embeddings"],
        all_split_assets["test"]["adata"],
    )

    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")

    # Apply dimensionality reduction if specified
    if snakemake.params.dimreduce == "PCA":
        from sklearn.decomposition import PCA

        latent_dim = snakemake.params.latent_dim
        logger.info(
            f"Performing PCA dimensionality reduction to {latent_dim} components"
        )

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "PCA",
                    PCA(n_components=latent_dim, random_state=snakemake.params.seed),
                ),
            ]
        )
        X_train = torch.Tensor(pipe.fit_transform(X_train))
        X_test = torch.Tensor(pipe.transform(X_test))

    # Run regression evaluation using HEST's trainer
    logger.info("Running regression evaluation...")
    probe_results, linprobe_dump = train_test_reg(
        X_train,
        X_test,
        y_train,
        y_test,
        random_state=snakemake.params.seed,
        genes=genes,
        method=snakemake.params.method,
    )

    # Create summary
    probe_summary = {"n_train": len(y_train), "n_test": len(y_test)}
    probe_summary.update({key: val for key, val in probe_results.items()})

    logger.info(
        f"Evaluation results: Pearson mean={probe_results['pearson_mean']:.4f} ± {probe_results['pearson_std']:.4f}"
    )

    # Save results
    with open(results_dir / "results.json", "w") as f:
        json.dump(probe_results, f, sort_keys=True, indent=4)

    with open(results_dir / "summary.json", "w") as f:
        json.dump(probe_summary, f, sort_keys=True, indent=4)

    save_pkl(results_dir / "inference_dump.pkl", linprobe_dump)

    return probe_results


def evaluate_dataset_folds(dataset_bench_path, embeddings_dir, save_dir):
    """
    Evaluate all folds for a given dataset using SpotWhisperer embeddings.

    This function is adapted from HEST's predict_folds.
    """

    splits_dir = dataset_bench_path / "splits"

    # Find all splits
    split_files = list(splits_dir.glob("*.csv"))
    train_splits = sorted([f for f in split_files if f.name.startswith("train_")])
    test_splits = sorted([f for f in split_files if f.name.startswith("test_")])

    if len(train_splits) != len(test_splits):
        raise ValueError(
            f"Mismatch in number of train ({len(train_splits)}) and test ({len(test_splits)}) splits"
        )

    n_splits = len(train_splits)

    # Evaluate each fold
    fold_results = []
    for i in range(n_splits):
        train_split_file = splits_dir / f"train_{i}.csv"
        test_split_file = splits_dir / f"test_{i}.csv"

        fold_save_dir = save_dir / f"split{i}"
        fold_save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Evaluating fold {i}/{n_splits-1}")

        try:
            fold_result = evaluate_single_split(
                train_split_file,
                test_split_file,
                dataset_bench_path,
                embeddings_dir,
                fold_save_dir,
            )
            fold_results.append(fold_result)
        except Exception as e:
            logger.error(f"Error evaluating fold {i}: {str(e)}")
            raise

    # Merge results across folds
    kfold_results = merge_fold_results(fold_results)

    # Save k-fold results
    with open(save_dir / "results_kfold.json", "w") as f:
        p_corrs = kfold_results["pearson_corrs"]
        p_corrs = sorted(p_corrs, key=lambda x: x["mean"], reverse=True)
        kfold_results["pearson_corrs"] = p_corrs
        json.dump(kfold_results, f, sort_keys=True, indent=4)

    logger.info(
        f"K-fold evaluation completed: Pearson mean={kfold_results['pearson_mean']:.4f} ± {kfold_results['pearson_std']:.4f}"
    )

    return kfold_results


# Extract dataset name from wildcards and use input/output paths directly
dataset_bench_path = Path(snakemake.input.dataset_dir)
embeddings_dir = Path(snakemake.input.embeddings_dir)

logger.info(f"Dataset bench path: {dataset_bench_path}")
logger.info(f"Embeddings dir: {embeddings_dir}")

# Use the output directory directly from snakemake
spotwhisperer_results_dir = Path(snakemake.output.results_dir)
spotwhisperer_results_dir.mkdir(parents=True, exist_ok=True)

logger.info(f"Results will be saved to: {spotwhisperer_results_dir}")

dataset_results = evaluate_dataset_folds(
    dataset_bench_path, embeddings_dir, spotwhisperer_results_dir
)

dataset_performance_summary = {
    "dataset_name": snakemake.wildcards.dataset,
    "pearson_mean": dataset_results["pearson_mean"],
    "pearson_std": dataset_results["pearson_std"],
}

# Save dataset-specific summary
with open(spotwhisperer_results_dir / "dataset_summary.json", "w") as f:
    json.dump(dataset_performance_summary, f, sort_keys=True, indent=4)

logger.info(
    f"performance: {dataset_results['pearson_mean']:.4f} ± {dataset_results['pearson_std']:.4f}"
)
logger.info("HEST evaluation completed successfully")
