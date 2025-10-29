#!/usr/bin/env python
"""
Debug script to investigate why the image-transcriptome model (hest1k)
is performing unreasonably well on text-transcriptome benchmarks.

This script checks:
1. Model mapping logic in Snakefile
2. File paths being used for each model
3. Actual content of aggregated results
4. Raw CellWhisperer test results
"""

import pandas as pd
import json
from pathlib import Path
import glob

# Define paths based on Snakefile structure
PROJECT_DIR = Path("../../..")  # Adjust based on actual project structure
BENCHMARKS_DIR = PROJECT_DIR / "results/spotwhisperer_eval/benchmarks"

# Model mappings from Snakefile
MODEL_MAPPINGS = {
    "cellxgene_census__archs4_geo": {
        "naive_baseline": "hest1k",
        "bimodal_matching": "cellxgene_census__archs4_geo",
        "bimodal_bridge": "hest1k__quilt1m",
        "trimodal": "cellxgene_census__archs4_geo__hest1k__quilt1m",
    },
    "hest1k": {
        "naive_baseline": "quilt1m",
        "bimodal_matching": "hest1k",
        "bimodal_bridge": "cellxgene_census__archs4_geo__quilt1m",
        "trimodal": "cellxgene_census__archs4_geo__hest1k__quilt1m",
    },
    "quilt1m": {
        "naive_baseline": "cellxgene_census__archs4_geo",
        "bimodal_matching": "quilt1m",
        "bimodal_bridge": "cellxgene_census__archs4_geo__hest1k",
        "trimodal": "cellxgene_census__archs4_geo__hest1k__quilt1m",
    },
}


def check_model_mappings():
    """Check that model mappings make sense"""
    print("=== MODEL MAPPING VERIFICATION ===")

    # Models used in spider plot
    spider_models = [
        ("trimodal", MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["trimodal"]),
        (
            "text-transcriptome",
            MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["bimodal_matching"],
        ),
        ("image-transcriptome", MODEL_MAPPINGS["hest1k"]["bimodal_matching"]),
        ("image-text", MODEL_MAPPINGS["quilt1m"]["bimodal_matching"]),
    ]

    for model_name, dataset_combo in spider_models:
        print(f"{model_name:20} -> {dataset_combo}")

    print("\nExpected:")
    print("text-transcriptome   -> cellxgene_census__archs4_geo (✓)")
    print("image-transcriptome  -> hest1k (✓)")
    print("image-text          -> quilt1m (✓)")
    print()


def check_aggregated_files():
    """Check the content of aggregated CellWhisperer results"""
    print("=== AGGREGATED CELLWHISPERER RESULTS ===")

    models_to_check = [
        ("text-transcriptome", "cellxgene_census__archs4_geo"),
        ("image-transcriptome", "hest1k"),
        ("image-text", "quilt1m"),
        ("trimodal", "cellxgene_census__archs4_geo__hest1k__quilt1m"),
    ]

    for model_name, dataset_combo in models_to_check:
        file_path = (
            BENCHMARKS_DIR / "retrieval" / dataset_combo / "aggregated_cwevals.csv"
        )
        print(f"\n{model_name} ({dataset_combo}):")
        print(f"File: {file_path}")

        if file_path.exists():
            df = pd.read_csv(file_path, index_col=0)
            print(f"Shape: {df.shape}")

            # Look for text-transcriptome metrics
            text_transcriptome_metrics = [
                col for col in df.index if "TabSap" in str(col)
            ]
            if text_transcriptome_metrics:
                print("Text-transcriptome metrics found:")
                for metric in text_transcriptome_metrics:
                    value = df.loc[metric].iloc[0] if len(df.columns) > 0 else "N/A"
                    print(f"  {metric}: {value}")
            else:
                print("No text-transcriptome metrics found")
        else:
            print("File does not exist!")


def check_raw_test_results():
    """Check raw CellWhisperer test results before aggregation"""
    print("\n=== RAW CELLWHISPERER TEST RESULTS ===")

    # Check if hest1k model was tested on cellxgene_census__archs4_geo
    csv_logs_pattern = (
        PROJECT_DIR
        / "lightning_logs"
        / "sweval___hest1k___cellxgene_census__archs4_geo"
        / "metrics.csv"
    )

    print(f"Looking for: {csv_logs_pattern}")

    if csv_logs_pattern.exists():
        print("Found raw results file!")
        df = pd.read_csv(csv_logs_pattern)
        print(f"Shape: {df.shape}")
        print("Columns:", list(df.columns))

        # Look for TabSap metrics
        tabsap_cols = [col for col in df.columns if "TabSap" in col]
        if tabsap_cols:
            print(f"\nTabSap metrics in raw file: {len(tabsap_cols)}")
            for col in tabsap_cols[:5]:  # Show first 5
                print(f"  {col}")

            # Check the last row (final results)
            print(f"\nLast row TabSap values:")
            for col in tabsap_cols[:3]:
                value = df[col].iloc[-1]
                print(f"  {col}: {value}")
        else:
            print("No TabSap metrics found in raw file")
    else:
        print("Raw results file not found!")

        # Try to find any sweval files for hest1k
        pattern = PROJECT_DIR / "lightning_logs" / "sweval___hest1k___*" / "metrics.csv"
        matching_files = glob.glob(str(pattern))
        print(f"Found {len(matching_files)} sweval files for hest1k:")
        for f in matching_files:
            print(f"  {f}")


def check_aggregation_logic():
    """Check the aggregation logic in aggregate_retrieval_results.py"""
    print("\n=== AGGREGATION LOGIC CHECK ===")

    # Simulate what happens in aggregate_retrieval_results.py
    print("The aggregation script does:")
    print("1. Loads metrics.csv from multiple test datasets")
    print("2. Takes the last row (final results) from each")
    print("3. For CellWhisperer metrics, takes iloc[0] assuming they're the same")
    print("4. This could be problematic if:")
    print("   - Different test datasets give different CW results")
    print("   - The 'same across runs' assumption is wrong")
    print("   - File paths are mixed up")


def main():
    """Run all debugging checks"""
    print(
        "DEBUGGING: Image-transcriptome model performing well on text-transcriptome tasks"
    )
    print("=" * 80)

    check_model_mappings()
    check_aggregated_files()
    check_raw_test_results()
    check_aggregation_logic()

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("1. Check if the aggregated_cwevals.csv files contain the expected data")
    print("2. Verify that hest1k model wasn't accidentally trained on text data")
    print("3. Check if there's a file path mixup in the spider plot")
    print("4. Look at the actual spider plot notebook to see data loading")


if __name__ == "__main__":
    main()
