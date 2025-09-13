#!/usr/bin/env python3
"""
Aggregate HEST benchmark results from all datasets into a single summary.

This script combines individual dataset results into an overall benchmark summary.
"""

import json
import numpy as np
from pathlib import Path
import logging
import pandas as pd

logger = logging.getLogger(__name__)


"""Aggregate results from all datasets"""

# Get summaries from snakemake input

aggregated_data = {
    "datasets": [],
    "overall_performance": 0.0,
    "n_datasets": 0,
    "performance_by_dataset": {},
}


for dataset_name, summary_file in zip(
    snakemake.params.datasets, snakemake.input.results_csvs
):

    logger.info(f"Loading results for dataset: {dataset_name}")

    with open(summary_file, "r") as f:
        dataset_summary = pd.read_csv(summary_file)

    dataset_performance = dataset_summary.iloc[-1][snakemake.params.metrics].to_dict()

    dataset_performance["dataset_name"] = dataset_name

    aggregated_data["datasets"].append(dataset_performance)
    aggregated_data["performance_by_dataset"][dataset_name] = dataset_performance

aggregated_data["overall_performance"] = np.mean(
    [d[snakemake.params.metrics[0]] for d in aggregated_data["datasets"]]
)

aggregated_data["n_datasets"] = len(snakemake.params.datasets)


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
