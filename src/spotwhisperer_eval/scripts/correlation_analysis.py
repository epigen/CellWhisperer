#!/usr/bin/env python3
"""
Correlation analysis between LLM scores and performance improvements
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict


def load_classifications() -> tuple[Dict, Dict]:
    """Load both histopathology and HEST representation classifications"""

    # Load histopathology classifications
    with open(snakemake.input.histopathology_classifications, "r") as f:
        histopathology_classifications = json.load(f)

    # Load HEST representation classifications
    with open(snakemake.input.hest_classifications, "r") as f:
        hest_classifications = json.load(f)

    return histopathology_classifications, hest_classifications


def load_performance_data() -> pd.DataFrame:
    """Load per-class performance analysis data"""

    df = pd.read_csv(snakemake.input.per_class_analysis)

    # Filter for human_disease dataset and merge F1/ROCAUC data
    human_disease_df = df[df["dataset"] == "human_disease"].copy()

    # Filter for rocauc and f1 metrics separately
    rocauc_data = human_disease_df[human_disease_df["metric"] == "rocauc"].copy()
    f1_data = human_disease_df[human_disease_df["metric"] == "f1"].copy()

    # Remove rows with NaN improvements
    rocauc_data = rocauc_data.dropna(subset=["rocauc_improvement"])
    f1_data = f1_data.dropna(subset=["f1_improvement"])

    # Merge rocauc and f1 data on class
    merged_data = pd.merge(
        rocauc_data[["class", "rocauc_improvement", "rocauc_relative_improvement"]],
        f1_data[["class", "f1_improvement", "f1_relative_improvement"]],
        on="class",
        how="inner",
    )

    return merged_data


def extract_disease_name(class_name: str) -> str:
    """Extract disease name from class description"""
    import re

    pattern = r"A sample of (.+?) from a healthy individual"
    match = re.search(pattern, class_name)
    if match:
        return match.group(1).lower().strip()
    return None


def create_correlation_analysis(
    performance_data: pd.DataFrame,
    histopathology_classifications: Dict,
    hest_classifications: Dict,
) -> pd.DataFrame:
    """Create merged dataset with all scores and performance metrics"""

    analysis_data = []

    for _, row in performance_data.iterrows():
        disease_name = extract_disease_name(row["class"])

        if (
            disease_name
            and disease_name in histopathology_classifications
            and disease_name in hest_classifications
        ):

            he_score = histopathology_classifications[disease_name]
            hest_score = hest_classifications[disease_name]
            combined_score = he_score + hest_score

            analysis_data.append(
                {
                    "disease_name": disease_name,
                    "class": row["class"],
                    "he_detectability": he_score,
                    "hest_representation": hest_score,
                    "combined_score": combined_score,
                    "f1_improvement": row["f1_improvement"],
                    "f1_relative_improvement": row["f1_relative_improvement"],
                    "rocauc_improvement": row["rocauc_improvement"],
                    "rocauc_relative_improvement": row["rocauc_relative_improvement"],
                }
            )

    return pd.DataFrame(analysis_data)


def create_correlation_plots(analysis_df: pd.DataFrame, output_dir: Path):
    """Create correlation plots for each score type vs performance metrics"""

    # Apply matplotlib style
    plt.style.use(snakemake.input.mpl_style)

    # Define the score types and their labels
    score_types = [
        ("he_detectability", "H&E Detectability Score"),
        ("hest_representation", "HEST Representation Score"),
        ("combined_score", "Combined Score (H&E + HEST)"),
    ]

    # Define performance metrics
    performance_metrics = [
        ("f1_improvement", "F1 Improvement"),
        ("f1_relative_improvement", "F1 Relative Improvement (%)"),
        ("rocauc_improvement", "ROCAUC Improvement"),
        ("rocauc_relative_improvement", "ROCAUC Relative Improvement (%)"),
    ]

    for score_col, score_label in score_types:
        # Create figure with 2x2 subplots for the 4 performance metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"{score_label} vs Performance Improvements", fontsize=16)

        for i, (perf_col, perf_label) in enumerate(performance_metrics):
            ax = axes[i // 2, i % 2]

            # Create scatter plot
            ax.scatter(
                analysis_df[score_col], analysis_df[perf_col], alpha=0.6, color="blue"
            )

            # Calculate correlation
            correlation = analysis_df[score_col].corr(analysis_df[perf_col])

            # Add trend line
            if not np.isnan(correlation) and len(analysis_df) > 2:
                z = np.polyfit(analysis_df[score_col], analysis_df[perf_col], 1)
                p = np.poly1d(z)
                ax.plot(
                    analysis_df[score_col],
                    p(analysis_df[score_col]),
                    "r--",
                    alpha=0.8,
                    label=f"r={correlation:.3f}",
                )
                ax.legend()

            # Set labels and title
            ax.set_xlabel(score_label)
            ax.set_ylabel(perf_label)
            ax.set_title(f"{perf_label} vs {score_label}")
            ax.grid(True, alpha=0.3)

            # Add correlation text
            ax.text(
                0.05,
                0.95,
                f"r = {correlation:.3f}",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                verticalalignment="top",
            )

        plt.tight_layout()

        # Save plot
        plot_filename = f"correlation_{score_col}_vs_performance.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved correlation plot: {plot_path}")

        plt.show()


def create_four_group_box_plots(analysis_df: pd.DataFrame, output_dir: Path):
    """Create box plots comparing four groups based on H&E and HEST scores"""

    # Apply matplotlib style
    plt.style.use(snakemake.input.mpl_style)

    # Create four groups based on score thresholds
    analysis_df = analysis_df.copy()

    def categorize_disease(row):
        he_high = row["he_detectability"] >= 6
        hest_high = row["hest_representation"] >= 6

        if he_high and hest_high:
            return "Both High (≥6)"
        elif he_high and not hest_high:
            return "H&E High Only"
        elif not he_high and hest_high:
            return "HEST High Only"
        else:
            return "Both Low (<6)"

    analysis_df["score_category"] = analysis_df.apply(categorize_disease, axis=1)

    # Define performance metrics
    performance_metrics = [
        ("f1_improvement", "F1 Improvement"),
        ("f1_relative_improvement", "F1 Relative Improvement (%)"),
        ("rocauc_improvement", "ROCAUC Improvement"),
        ("rocauc_relative_improvement", "ROCAUC Relative Improvement (%)"),
    ]

    # Create figure with 2x2 subplots for the 4 performance metrics
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(
        "Performance Improvements by H&E Detectability and HEST Representation Groups",
        fontsize=16,
    )

    # Define colors for each group
    group_colors = {
        "Both High (≥6)": "#2ecc71",  # Green
        "H&E High Only": "#3498db",  # Blue
        "HEST High Only": "#e74c3c",  # Red
        "Both Low (<6)": "#95a5a6",  # Gray
    }

    for i, (perf_col, perf_label) in enumerate(performance_metrics):
        ax = axes[i // 2, i % 2]

        # Create box plot
        sns.boxplot(
            data=analysis_df,
            x="score_category",
            y=perf_col,
            ax=ax,
            palette=group_colors,
        )

        ax.set_xlabel("Score Category")
        ax.set_ylabel(perf_label)
        ax.set_title(f"{perf_label} by Score Groups")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

        # Add sample sizes for each group
        for j, category in enumerate(
            ["Both High (≥6)", "H&E High Only", "HEST High Only", "Both Low (<6)"]
        ):
            count = (analysis_df["score_category"] == category).sum()
            if count > 0:
                y_max = ax.get_ylim()[1]
                ax.text(
                    j,
                    y_max * 0.95,
                    f"n={count}",
                    ha="center",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                )

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "four_group_box_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved four-group box plots: {plot_path}")

    # Print group statistics
    print(f"\n=== FOUR-GROUP ANALYSIS ===")
    print(f"Group distribution:")
    for category in [
        "Both High (≥6)",
        "H&E High Only",
        "HEST High Only",
        "Both Low (<6)",
    ]:
        count = (analysis_df["score_category"] == category).sum()
        percentage = count / len(analysis_df) * 100
        print(f"  {category}: {count} diseases ({percentage:.1f}%)")

    print(f"\nGroup performance statistics:")
    for category in [
        "Both High (≥6)",
        "H&E High Only",
        "HEST High Only",
        "Both Low (<6)",
    ]:
        category_data = analysis_df[analysis_df["score_category"] == category]
        if len(category_data) > 0:
            print(f"\n{category} (n={len(category_data)}):")
            for perf_col, perf_label in performance_metrics:
                values = category_data[perf_col]
                print(f"  {perf_label}:")
                print(f"    Mean: {values.mean():.4f}")
                print(f"    Median: {values.median():.4f}")
                print(f"    Std: {values.std():.4f}")

    plt.show()

    return analysis_df


def print_correlation_summary(analysis_df: pd.DataFrame):
    """Print summary of all correlations"""

    score_types = [
        ("he_detectability", "H&E Detectability"),
        ("hest_representation", "HEST Representation"),
        ("combined_score", "Combined Score"),
    ]

    performance_metrics = [
        ("f1_improvement", "F1 Improvement"),
        ("f1_relative_improvement", "F1 Relative Improvement"),
        ("rocauc_improvement", "ROCAUC Improvement"),
        ("rocauc_relative_improvement", "ROCAUC Relative Improvement"),
    ]

    print("\n=== CORRELATION SUMMARY ===")
    print(f"Total diseases analyzed: {len(analysis_df)}")

    for score_col, score_label in score_types:
        print(f"\n{score_label}:")
        for perf_col, perf_label in performance_metrics:
            correlation = analysis_df[score_col].corr(analysis_df[perf_col])
            print(f"  vs {perf_label}: r={correlation:.4f}")

    # Print score distributions
    print(f"\n=== SCORE DISTRIBUTIONS ===")
    for score_col, score_label in score_types:
        values = analysis_df[score_col]
        print(f"\n{score_label}:")
        print(f"  Mean: {values.mean():.2f}")
        print(f"  Std: {values.std():.2f}")
        print(f"  Range: {values.min():.0f} - {values.max():.0f}")


def main():
    """Main analysis function"""

    print("Loading classification data...")
    try:
        histopathology_classifications, hest_classifications = load_classifications()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Loading performance data...")
    try:
        performance_data = load_performance_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Creating correlation analysis...")
    analysis_df = create_correlation_analysis(
        performance_data, histopathology_classifications, hest_classifications
    )

    if len(analysis_df) == 0:
        print("No overlapping diseases found between all datasets!")
        return

    print(f"Found {len(analysis_df)} diseases with complete data")

    # Create output directory
    output_dir = Path(snakemake.output.analysis_data).parent

    # Create correlation plots
    create_correlation_plots(analysis_df, output_dir)

    # Create four-group box plots
    analysis_df_with_groups = create_four_group_box_plots(analysis_df, output_dir)

    # Save analysis data
    analysis_df_with_groups.to_csv(snakemake.output.analysis_data, index=False)
    print(f"Analysis data saved to: {snakemake.output.analysis_data}")

    # Print summary
    print_correlation_summary(analysis_df)


if __name__ == "__main__":
    main()
