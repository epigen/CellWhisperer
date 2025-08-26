#!/usr/bin/env python3
"""
Aggregate HEST benchmark results from all datasets into a single summary.

This script combines individual dataset results into an overall benchmark summary.
"""

import json
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def main():
    """Aggregate results from all datasets"""

    # Get summaries from snakemake input

    aggregated_data = {
        "datasets": [],
        "overall_performance": 0.0,
        "n_datasets": 0,
        "performance_by_dataset": {},
    }

    dataset_performances = []

    for dataset_name, summary_dir in zip(
        snakemake.params.datasets, snakemake.input.results_dirs
    ):
        summary_file = Path(summary_dir) / "dataset_summary.json"

        logger.info(f"Loading results for dataset: {dataset_name}")

        with open(summary_file, "r") as f:
            dataset_summary = json.load(f)

        # Extract performance metrics (looking for dataset_summary.json structure)
        pearson_mean = dataset_summary["pearson_mean"]
        pearson_std = dataset_summary["pearson_std"]

        dataset_performance = {
            "dataset_name": dataset_name,
            "pearson_mean": float(pearson_mean),
            "pearson_std": float(pearson_std),
        }

        aggregated_data["datasets"].append(dataset_performance)
        aggregated_data["performance_by_dataset"][dataset_name] = dataset_performance
        dataset_performances.append(pearson_mean)

        logger.info(f"Dataset {dataset_name}: {pearson_mean:.4f} ± {pearson_std:.4f}")

    # Calculate overall performance
    if dataset_performances:
        overall_performance = np.mean(dataset_performances)
        overall_std = np.std(dataset_performances)

        aggregated_data["overall_performance"] = float(overall_performance)
        aggregated_data["overall_std"] = float(overall_std)
        aggregated_data["n_datasets"] = len(dataset_performances)

        # Sort datasets by performance
        aggregated_data["datasets"].sort(key=lambda x: x["pearson_mean"], reverse=True)

        logger.info(
            f"Overall SpotWhisperer performance: {overall_performance:.4f} ± {overall_std:.4f}"
        )
        logger.info(
            f"Successfully aggregated results from {len(dataset_performances)} datasets"
        )
    else:
        logger.error("No valid dataset results found")
        aggregated_data["overall_performance"] = 0.0
        aggregated_data["overall_std"] = 0.0
        aggregated_data["n_datasets"] = 0

    # Add metadata
    aggregated_data["metadata"] = {
        "model": snakemake.wildcards.model,
        "benchmark": "HEST",
        "aggregation_method": "arithmetic_mean",
        "num_datasets": len(snakemake.params.datasets),
    }

    # Save aggregated results
    with open(snakemake.output.aggregated_summary, "w") as f:
        json.dump(aggregated_data, f, sort_keys=True, indent=4)

    logger.info(f"Aggregated results saved to: {snakemake.output.aggregated_summary}")


if __name__ == "__main__":
    main()
