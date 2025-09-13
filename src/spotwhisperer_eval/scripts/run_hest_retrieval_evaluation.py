#!/usr/bin/env python3
"""
Run HEST retrieval evaluation using SpotWhisperer embeddings.

This script uses the pre-computed SpotWhisperer embeddings (both image and transcriptome)
to perform retrieval-based evaluation using the existing CellWhisperer evaluation functions.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

# Import CellWhisperer retrieval evaluation functions
from cellwhisperer.validation.zero_shot.functions import (
    get_performance_metrics_left_vs_right,
)
from cellwhisperer.utils.model_io import load_cellwhisperer_model
from hest.bench.utils.file_utils import read_assets_from_h5


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_embeddings_from_h5(embed_path):
    """Load both image and transcriptome embeddings from H5 file"""
    assets, _ = read_assets_from_h5(str(embed_path))

    # Load image embeddings (backward compatibility)
    if "image_embeddings" in assets:
        image_embeds = assets["image_embeddings"]
    else:
        image_embeds = assets["embeddings"]  # fallback to legacy key

    # Load transcriptome embeddings if available
    transcriptome_embeds = assets.get("transcriptome_embeddings", None)
    barcodes = assets["barcodes"].flatten().astype(str).tolist()

    return image_embeds, transcriptome_embeds, barcodes


def evaluate_single_split(
    test_split_file, dataset_bench_path, embeddings_dir, results_dir, model
):
    """
    Evaluate a single split using SpotWhisperer embeddings for retrieval.

    Uses the CellWhisperer retrieval evaluation functions directly.
    """
    # Read test split file
    test_df = pd.read_csv(test_split_file)
    logger.info(f"Processing {len(test_df)} test samples")

    # Collect all embeddings
    all_image_embeds = []
    all_transcriptome_embeds = []
    all_sample_ids = []

    for i in tqdm(range(len(test_df)), desc="Loading embeddings"):
        sample_id = test_df.iloc[i]["sample_id"]
        embed_path = embeddings_dir / f"{sample_id}.h5"

        if not embed_path.exists():
            logger.warning(f"Embedding file not found: {embed_path}")
            continue

        # Load embeddings
        image_embeds, transcriptome_embeds, barcodes = load_embeddings_from_h5(
            embed_path
        )

        if transcriptome_embeds is None:
            logger.error(f"No transcriptome embeddings found for {sample_id}")
            logger.error(
                "Please regenerate embeddings using the updated inference script"
            )
            raise ValueError("Missing transcriptome embeddings")

        # Store embeddings for this sample
        all_image_embeds.append(image_embeds)
        all_transcriptome_embeds.append(transcriptome_embeds)
        all_sample_ids.extend([sample_id] * len(image_embeds))

    # Concatenate all embeddings
    if not all_image_embeds:
        raise ValueError("No embeddings loaded")

    image_embeds_tensor = torch.tensor(np.concatenate(all_image_embeds, axis=0))
    transcriptome_embeds_tensor = torch.tensor(
        np.concatenate(all_transcriptome_embeds, axis=0)
    )

    logger.info(
        f"Loaded embeddings: {image_embeds_tensor.shape} image, {transcriptome_embeds_tensor.shape} transcriptome"
    )

    # Create correct matching indices (each spot matches to itself)
    n_spots = len(image_embeds_tensor)
    correct_indices = list(range(n_spots))

    # Run retrieval evaluation using CellWhisperer functions
    logger.info("Running retrieval evaluation...")

    # Image -> Transcriptome retrieval
    img_to_transcriptome_metrics, img_to_transcriptome_df = (
        get_performance_metrics_left_vs_right(
            model=model,
            left_input=image_embeds_tensor,
            right_input=transcriptome_embeds_tensor,
            correct_right_idx_per_left=correct_indices,
            average_mode=None,  # No averaging, evaluate at spot level
            batch_size=128,
            report_per_class_metrics=False,
            right_as_classes=True,
        )
    )

    # Transcriptome -> Image retrieval
    transcriptome_to_img_metrics, transcriptome_to_img_df = (
        get_performance_metrics_left_vs_right(
            model=model,
            left_input=transcriptome_embeds_tensor,
            right_input=image_embeds_tensor,
            correct_right_idx_per_left=correct_indices,
            average_mode=None,  # No averaging, evaluate at spot level
            batch_size=128,
            report_per_class_metrics=False,
            right_as_classes=True,
        )
    )

    # Extract key metrics
    results = {
        "n_spots": n_spots,
        "img_to_transcriptome_recall@1": float(
            img_to_transcriptome_metrics["recall_1"]
        ),
        "img_to_transcriptome_recall@5": float(
            img_to_transcriptome_metrics["recall_5"]
        ),
        "img_to_transcriptome_recall@10": float(
            img_to_transcriptome_metrics["recall_10"]
        ),
        "img_to_transcriptome_recall@50": float(
            img_to_transcriptome_metrics["recall_50"]
        ),
        "img_to_transcriptome_auroc": float(img_to_transcriptome_metrics["auroc"]),
        "transcriptome_to_img_recall@1": float(
            transcriptome_to_img_metrics["recall_1"]
        ),
        "transcriptome_to_img_recall@5": float(
            transcriptome_to_img_metrics["recall_5"]
        ),
        "transcriptome_to_img_recall@10": float(
            transcriptome_to_img_metrics["recall_10"]
        ),
        "transcriptome_to_img_recall@50": float(
            transcriptome_to_img_metrics["recall_50"]
        ),
        "transcriptome_to_img_auroc": float(transcriptome_to_img_metrics["auroc"]),
    }

    logger.info(
        f"Results: img->transcriptome recall@1={results['img_to_transcriptome_recall@1']:.4f}, "
        f"transcriptome->img recall@1={results['transcriptome_to_img_recall@1']:.4f}"
    )

    # Save results
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, sort_keys=True, indent=4)

    # Save detailed results
    img_to_transcriptome_df.to_csv(
        results_dir / "img_to_transcriptome_detailed.csv", index=False
    )
    transcriptome_to_img_df.to_csv(
        results_dir / "transcriptome_to_img_detailed.csv", index=False
    )

    return results


def evaluate_dataset_folds(dataset_bench_path, embeddings_dir, save_dir, model):
    """
    Evaluate all folds for a given dataset using SpotWhisperer embeddings.
    """
    splits_dir = dataset_bench_path / "splits"

    # Find test splits (we only need test splits for retrieval evaluation)
    split_files = list(splits_dir.glob("*.csv"))
    test_splits = sorted([f for f in split_files if f.name.startswith("test_")])

    if not test_splits:
        raise ValueError("No test splits found")

    logger.info(f"Found {len(test_splits)} test splits")

    # Evaluate each fold
    fold_results = []
    for i, test_split_file in enumerate(test_splits):
        fold_save_dir = save_dir / f"split{i}"
        fold_save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Evaluating fold {i}/{len(test_splits)-1}")

        try:
            fold_result = evaluate_single_split(
                test_split_file,
                dataset_bench_path,
                embeddings_dir,
                fold_save_dir,
                model,
            )
            fold_results.append(fold_result)
        except Exception as e:
            logger.error(f"Error evaluating fold {i}: {str(e)}")
            raise

    # Aggregate results across folds
    if not fold_results:
        raise ValueError("No fold results obtained")

    # Compute mean and std for each metric
    aggregated_results = {}
    metric_keys = [k for k in fold_results[0].keys() if k != "n_spots"]

    for key in metric_keys:
        values = [result[key] for result in fold_results]
        aggregated_results[f"{key}_mean"] = np.mean(values)
        aggregated_results[f"{key}_std"] = np.std(values)
        aggregated_results[f"{key}_values"] = values

    # Add sample count info
    aggregated_results["total_spots"] = sum(
        result["n_spots"] for result in fold_results
    )
    aggregated_results["n_folds"] = len(fold_results)

    # Save k-fold results
    with open(save_dir / "results_kfold.json", "w") as f:
        json.dump(aggregated_results, f, sort_keys=True, indent=4)

    logger.info(
        f"K-fold evaluation completed: img->transcriptome recall@1={aggregated_results['img_to_transcriptome_recall@1_mean']:.4f}±{aggregated_results['img_to_transcriptome_recall@1_std']:.4f}, "
        f"transcriptome->img recall@1={aggregated_results['transcriptome_to_img_recall@1_mean']:.4f}±{aggregated_results['transcriptome_to_img_recall@1_std']:.4f}"
    )

    return aggregated_results


# Main execution
if __name__ == "__main__":
    # Extract paths from snakemake
    dataset_bench_path = Path(snakemake.input.dataset_dir)
    embeddings_dir = Path(snakemake.input.embeddings_dir)
    model_path = snakemake.input.model

    logger.info(f"Dataset bench path: {dataset_bench_path}")
    logger.info(f"Embeddings dir: {embeddings_dir}")
    logger.info(f"Model path: {model_path}")

    # Use the output directory directly from snakemake
    spotwhisperer_results_dir = Path(snakemake.output.results_dir)
    spotwhisperer_results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Results will be saved to: {spotwhisperer_results_dir}")

    # Load SpotWhisperer model
    logger.info("Loading SpotWhisperer model...")
    (
        pl_model,
        text_processor,
        transcriptome_processor,
        image_processor,
    ) = load_cellwhisperer_model(model_path=model_path, eval=True)
    model = pl_model.model

    # Run evaluation
    dataset_results = evaluate_dataset_folds(
        dataset_bench_path, embeddings_dir, spotwhisperer_results_dir, model
    )

    # Create dataset performance summary
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
    logger.info("HEST retrieval evaluation completed successfully")
