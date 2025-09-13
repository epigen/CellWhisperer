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
import logging
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from hest.bench.st_dataset import load_adata
from hest.bench.utils.file_utils import read_assets_from_h5, save_pkl
from hest.bench.utils.utils import merge_dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_path(path):
    """Convert relative path to absolute path"""
    if path.startswith("./"):
        new_path = os.path.abspath(os.path.join(".", path))
    else:
        new_path = path
    return new_path


def compute_retrieval_metrics(
    image_embeds, transcriptome_embeds, k_values=[1, 5, 10, 50]
):
    """
    Compute retrieval metrics for both directions:
    - Image -> Transcriptome
    - Transcriptome -> Image

    Args:
        image_embeds: numpy array of shape (N, D) - image embeddings
        transcriptome_embeds: numpy array of shape (N, D) - transcriptome embeddings
        k_values: list of k values for recall@k computation

    Returns:
        dict: Dictionary containing all retrieval metrics
    """
    # Normalize embeddings for cosine similarity
    image_norm = image_embeds / np.linalg.norm(image_embeds, axis=1, keepdims=True)
    transcriptome_norm = transcriptome_embeds / np.linalg.norm(
        transcriptome_embeds, axis=1, keepdims=True
    )

    # Compute similarity matrix: (N_images, N_transcriptomes)
    similarity_matrix = np.dot(image_norm, transcriptome_norm.T)

    results = {}
    N = len(image_embeds)

    # Direction 1: Image -> Transcriptome retrieval
    img_to_transcriptome_recall = {}
    img_to_transcriptome_ranks = []

    for i in range(N):
        # Get similarities for this image query
        similarities = similarity_matrix[i, :]
        # Rank transcriptomes by similarity (descending order)
        ranked_indices = np.argsort(-similarities)
        # Find rank of correct transcriptome (ground truth is index i)
        correct_rank = np.where(ranked_indices == i)[0][0] + 1  # 1-indexed rank
        img_to_transcriptome_ranks.append(correct_rank)

    # Compute recall@k for image -> transcriptome
    for k in k_values:
        recall_k = np.mean([rank <= k for rank in img_to_transcriptome_ranks])
        img_to_transcriptome_recall[f"recall@{k}"] = recall_k

    # Compute AUROC for image -> transcriptome
    img_to_transcriptome_scores = []
    img_to_transcriptome_labels = []
    for i in range(N):
        similarities = similarity_matrix[i, :]
        # Binary labels: 1 for correct match, 0 for others
        labels = np.zeros(N)
        labels[i] = 1
        img_to_transcriptome_scores.extend(similarities)
        img_to_transcriptome_labels.extend(labels)

    img_to_transcriptome_auroc = roc_auc_score(
        img_to_transcriptome_labels, img_to_transcriptome_scores
    )

    # Direction 2: Transcriptome -> Image retrieval
    transcriptome_to_img_recall = {}
    transcriptome_to_img_ranks = []

    # Transpose similarity matrix for transcriptome queries
    transcriptome_similarity_matrix = (
        similarity_matrix.T
    )  # (N_transcriptomes, N_images)

    for i in range(N):
        # Get similarities for this transcriptome query
        similarities = transcriptome_similarity_matrix[i, :]
        # Rank images by similarity (descending order)
        ranked_indices = np.argsort(-similarities)
        # Find rank of correct image (ground truth is index i)
        correct_rank = np.where(ranked_indices == i)[0][0] + 1  # 1-indexed rank
        transcriptome_to_img_ranks.append(correct_rank)

    # Compute recall@k for transcriptome -> image
    for k in k_values:
        recall_k = np.mean([rank <= k for rank in transcriptome_to_img_ranks])
        transcriptome_to_img_recall[f"recall@{k}"] = recall_k

    # Compute AUROC for transcriptome -> image
    transcriptome_to_img_scores = []
    transcriptome_to_img_labels = []
    for i in range(N):
        similarities = transcriptome_similarity_matrix[i, :]
        # Binary labels: 1 for correct match, 0 for others
        labels = np.zeros(N)
        labels[i] = 1
        transcriptome_to_img_scores.extend(similarities)
        transcriptome_to_img_labels.extend(labels)

    transcriptome_to_img_auroc = roc_auc_score(
        transcriptome_to_img_labels, transcriptome_to_img_scores
    )

    # Combine results
    results.update(
        {
            "img_to_transcriptome_" + key: value
            for key, value in img_to_transcriptome_recall.items()
        }
    )
    results.update(
        {
            "transcriptome_to_img_" + key: value
            for key, value in transcriptome_to_img_recall.items()
        }
    )
    results["img_to_transcriptome_auroc"] = img_to_transcriptome_auroc
    results["transcriptome_to_img_auroc"] = transcriptome_to_img_auroc

    # Add mean reciprocal rank for additional insight
    results["img_to_transcriptome_mrr"] = np.mean(
        [1.0 / rank for rank in img_to_transcriptome_ranks]
    )
    results["transcriptome_to_img_mrr"] = np.mean(
        [1.0 / rank for rank in transcriptome_to_img_ranks]
    )

    return results


def merge_fold_results(results_arr):
    """
    Merge results from multiple folds for retrieval metrics
    """
    if not results_arr:
        return {}

    # Get all metric keys from first result
    metric_keys = list(results_arr[0].keys())

    # Aggregate metrics across folds
    aggregated_results = {}
    for key in metric_keys:
        if key in ["n_train", "n_test"]:
            # For count metrics, just take the first value (should be same across folds)
            aggregated_results[key] = results_arr[0][key]
        else:
            # For performance metrics, compute mean and std across folds
            values = [result[key] for result in results_arr]
            aggregated_results[f"{key}_mean"] = np.mean(values)
            aggregated_results[f"{key}_std"] = np.std(values)
            aggregated_results[f"{key}_values"] = values

    return aggregated_results


def evaluate_single_split(
    train_split_file, test_split_file, dataset_bench_path, embeddings_dir, results_dir
):
    """
    Evaluate a single train/test split using SpotWhisperer embeddings for retrieval.

    For retrieval evaluation, we only use the test split since no training is needed.
    We compute retrieval metrics between image embeddings and transcriptome embeddings.
    """
    # Read split files (we'll primarily use test split for retrieval)
    test_df = pd.read_csv(test_split_file)

    logger.info(f"Loading embeddings from: {embeddings_dir}")

    with open(Path(snakemake.input.dataset_dir) / "var_50genes.json", "r") as f:
        genes = json.load(f)["genes"]

    logger.info(f"Using {len(genes)} genes for evaluation")

    # Load test data for retrieval evaluation
    test_assets = {}
    logger.info(f"Loading test split data for retrieval evaluation...")

    for i in tqdm(range(len(test_df)), desc="Loading test samples"):
        sample_id = test_df.iloc[i]["sample_id"]
        embed_path = embeddings_dir / f"{sample_id}.h5"
        expr_path = dataset_bench_path / test_df.iloc[i]["expr_path"]

        # Load embeddings and metadata
        assets, _ = read_assets_from_h5(str(embed_path))
        barcodes = assets["barcodes"].flatten().astype(str).tolist()

        # Load expression data (this serves as our "transcriptome embedding")
        adata = load_adata(
            str(expr_path),
            genes=genes,
            barcodes=barcodes,
            normalize=snakemake.params.normalize,
        )
        assets["adata"] = adata.values

        # Merge assets
        test_assets = merge_dict(test_assets, assets)

    # Concatenate all test data
    if test_assets:
        for key, val in test_assets.items():
            test_assets[key] = np.concatenate(val, axis=0)

        logger.info(
            f"Loaded test split with {len(test_assets['embeddings'])} samples: {test_assets['embeddings'].shape}"
        )
    else:
        raise ValueError(f"No test data loaded")

    # Extract embeddings and transcriptome data
    image_embeds = test_assets["embeddings"]  # SpotWhisperer image embeddings
    transcriptome_embeds = test_assets[
        "adata"
    ]  # Gene expression as transcriptome embedding

    logger.info(f"Image embeddings shape: {image_embeds.shape}")
    logger.info(f"Transcriptome embeddings shape: {transcriptome_embeds.shape}")

    # Apply dimensionality reduction if specified
    if snakemake.params.dimreduce == "PCA":
        from sklearn.decomposition import PCA

        latent_dim = snakemake.params.latent_dim
        logger.info(
            f"Performing PCA dimensionality reduction to {latent_dim} components"
        )

        # Apply PCA to image embeddings
        image_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "PCA",
                    PCA(n_components=latent_dim, random_state=snakemake.params.seed),
                ),
            ]
        )
        image_embeds = image_pipe.fit_transform(image_embeds)

        # Apply PCA to transcriptome embeddings
        transcriptome_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "PCA",
                    PCA(n_components=latent_dim, random_state=snakemake.params.seed),
                ),
            ]
        )
        transcriptome_embeds = transcriptome_pipe.fit_transform(transcriptome_embeds)

    # Run retrieval evaluation
    logger.info("Running retrieval evaluation...")
    retrieval_results = compute_retrieval_metrics(
        image_embeds, transcriptome_embeds, k_values=[1, 5, 10, 50]
    )

    # Create summary
    retrieval_summary = {"n_test": len(image_embeds)}
    retrieval_summary.update({key: val for key, val in retrieval_results.items()})

    logger.info(
        f"Retrieval results: img->transcriptome recall@1={retrieval_results['img_to_transcriptome_recall@1']:.4f}, "
        f"transcriptome->img recall@1={retrieval_results['transcriptome_to_img_recall@1']:.4f}"
    )

    # Save results
    with open(results_dir / "results.json", "w") as f:
        json.dump(retrieval_results, f, sort_keys=True, indent=4)

    with open(results_dir / "summary.json", "w") as f:
        json.dump(retrieval_summary, f, sort_keys=True, indent=4)

    return retrieval_results


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
        json.dump(kfold_results, f, sort_keys=True, indent=4)

    # Log key retrieval metrics
    img_to_transcriptome_recall1_mean = kfold_results.get(
        "img_to_transcriptome_recall@1_mean", 0
    )
    transcriptome_to_img_recall1_mean = kfold_results.get(
        "transcriptome_to_img_recall@1_mean", 0
    )

    logger.info(
        f"K-fold evaluation completed: img->transcriptome recall@1={img_to_transcriptome_recall1_mean:.4f}, "
        f"transcriptome->img recall@1={transcriptome_to_img_recall1_mean:.4f}"
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
    "img_to_transcriptome_recall@1_mean": dataset_results.get(
        "img_to_transcriptome_recall@1_mean", 0
    ),
    "img_to_transcriptome_recall@1_std": dataset_results.get(
        "img_to_transcriptome_recall@1_std", 0
    ),
    "transcriptome_to_img_recall@1_mean": dataset_results.get(
        "transcriptome_to_img_recall@1_mean", 0
    ),
    "transcriptome_to_img_recall@1_std": dataset_results.get(
        "transcriptome_to_img_recall@1_std", 0
    ),
    "img_to_transcriptome_auroc_mean": dataset_results.get(
        "img_to_transcriptome_auroc_mean", 0
    ),
    "transcriptome_to_img_auroc_mean": dataset_results.get(
        "transcriptome_to_img_auroc_mean", 0
    ),
}

# Save dataset-specific summary
with open(spotwhisperer_results_dir / "dataset_summary.json", "w") as f:
    json.dump(dataset_performance_summary, f, sort_keys=True, indent=4)

img_to_transcriptome_recall1_mean = dataset_results.get(
    "img_to_transcriptome_recall@1_mean", 0
)
transcriptome_to_img_recall1_mean = dataset_results.get(
    "transcriptome_to_img_recall@1_mean", 0
)

logger.info(
    f"Final performance: img->transcriptome recall@1={img_to_transcriptome_recall1_mean:.4f}, "
    f"transcriptome->img recall@1={transcriptome_to_img_recall1_mean:.4f}"
)
logger.info("HEST evaluation completed successfully")
