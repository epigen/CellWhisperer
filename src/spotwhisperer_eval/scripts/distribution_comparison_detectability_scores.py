#!/usr/bin/env python3
"""
Distribution comparison of detectability scores for top improving diseases
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_data():
    """Load the required datasets"""
    # Load per-class analysis data
    per_class_df = pd.read_csv(snakemake.input.per_class_analysis)

    # Load HEST representation analysis
    hest_repr_df = pd.read_csv(snakemake.input.hest_representation_analysis)

    return per_class_df, hest_repr_df


def extract_disease_name(class_name):
    """Extract disease name from class description"""
    import re

    pattern = r"A sample of (.+?) from a healthy individual"
    match = re.search(pattern, class_name)
    if match:
        return match.group(1).lower().strip()
    return None


def prepare_analysis_data(per_class_df, hest_repr_df):
    """Prepare merged dataset with performance and detectability scores"""
    # Filter for human_disease dataset
    human_disease_df = per_class_df[per_class_df["dataset"] == "human_disease"].copy()

    # Filter for rocauc and f1 metrics separately
    rocauc_data = human_disease_df[human_disease_df["metric"] == "rocauc"].copy()
    f1_data = human_disease_df[human_disease_df["metric"] == "f1"].copy()

    # Remove rows with NaN improvements
    rocauc_data = rocauc_data.dropna(subset=["rocauc_improvement"])
    f1_data = f1_data.dropna(subset=["f1_improvement"])

    # Merge rocauc and f1 data on class
    merged_data = pd.merge(
        rocauc_data[["class", "rocauc_improvement"]],
        f1_data[["class", "f1_improvement"]],
        on="class",
        how="inner",
    )

    # Extract disease names and merge with detectability scores
    analysis_data = []
    for _, row in merged_data.iterrows():
        disease_name = extract_disease_name(row["class"])
        if disease_name:
            # Find matching disease in HEST representation data
            hest_match = hest_repr_df[hest_repr_df["disease_name"] == disease_name]
            if len(hest_match) > 0:
                analysis_data.append(
                    {
                        "class": row["class"],
                        "disease_name": disease_name,
                        "f1_improvement": row["f1_improvement"],
                        "rocauc_improvement": row["rocauc_improvement"],
                        "he_detectability": hest_match.iloc[0]["he_detectability"],
                        "hest_representation": hest_match.iloc[0][
                            "hest_representation"
                        ],
                    }
                )

    return pd.DataFrame(analysis_data)


def create_distribution_plots(analysis_df):
    """Create distribution comparison plots for top improving diseases"""

    # Apply matplotlib style
    plt.style.use(snakemake.input.mpl_style)

    # Get top 15 diseases for each metric
    top_f1_diseases = analysis_df.nlargest(15, "f1_improvement")
    top_rocauc_diseases = analysis_df.nlargest(15, "rocauc_improvement")

    print(f"Top 15 F1 improving diseases:")
    for _, row in top_f1_diseases.iterrows():
        print(f"  {row['disease_name']}: {row['f1_improvement']:.4f}")

    print(f"\nTop 15 ROCAUC improving diseases:")
    for _, row in top_rocauc_diseases.iterrows():
        print(f"  {row['disease_name']}: {row['rocauc_improvement']:.4f}")

    # Create the plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Detectability Score Distributions for Top Improving Diseases", fontsize=16
    )

    # Prepare data for violin plots
    def prepare_violin_data(top_diseases, score_col, metric_name):
        """Prepare data for violin plots comparing top diseases vs all diseases"""
        violin_data = []

        # Add background distribution (all diseases)
        for _, row in analysis_df.iterrows():
            violin_data.append(
                {
                    "group": f"All diseases (n={len(analysis_df)})",
                    "score": row[score_col],
                    "metric": metric_name,
                }
            )

        # Add top diseases distribution
        for _, row in top_diseases.iterrows():
            violin_data.append(
                {
                    "group": f"Top 15 {metric_name} improving (n=15)",
                    "score": row[score_col],
                    "metric": metric_name,
                }
            )

        return pd.DataFrame(violin_data)

    # Plot 1: H&E detectability for top F1 improving diseases
    f1_he_data = prepare_violin_data(top_f1_diseases, "he_detectability", "F1")
    sns.violinplot(data=f1_he_data, x="group", y="score", ax=axes[0, 0], inner="box")
    axes[0, 0].set_title("H&E Detectability Scores\n(Top F1 Improving vs All Diseases)")
    axes[0, 0].set_ylabel("H&E Detectability Score")
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: HEST representation for top F1 improving diseases
    f1_hest_data = prepare_violin_data(top_f1_diseases, "hest_representation", "F1")
    sns.violinplot(data=f1_hest_data, x="group", y="score", ax=axes[0, 1], inner="box")
    axes[0, 1].set_title(
        "HEST Representation Scores\n(Top F1 Improving vs All Diseases)"
    )
    axes[0, 1].set_ylabel("HEST Representation Score")
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: H&E detectability for top ROCAUC improving diseases
    rocauc_he_data = prepare_violin_data(
        top_rocauc_diseases, "he_detectability", "ROCAUC"
    )
    sns.violinplot(
        data=rocauc_he_data, x="group", y="score", ax=axes[1, 0], inner="box"
    )
    axes[1, 0].set_title(
        "H&E Detectability Scores\n(Top ROCAUC Improving vs All Diseases)"
    )
    axes[1, 0].set_ylabel("H&E Detectability Score")
    axes[1, 0].set_xlabel("")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: HEST representation for top ROCAUC improving diseases
    rocauc_hest_data = prepare_violin_data(
        top_rocauc_diseases, "hest_representation", "ROCAUC"
    )
    sns.violinplot(
        data=rocauc_hest_data, x="group", y="score", ax=axes[1, 1], inner="box"
    )
    axes[1, 1].set_title(
        "HEST Representation Scores\n(Top ROCAUC Improving vs All Diseases)"
    )
    axes[1, 1].set_ylabel("HEST Representation Score")
    axes[1, 1].set_xlabel("")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plt.savefig(snakemake.output.distribution_plots, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {snakemake.output.distribution_plots}")

    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")

    print(f"\nTop 15 F1 improving diseases:")
    print(
        f"  H&E detectability - Mean: {top_f1_diseases['he_detectability'].mean():.2f}, Std: {top_f1_diseases['he_detectability'].std():.2f}"
    )
    print(
        f"  HEST representation - Mean: {top_f1_diseases['hest_representation'].mean():.2f}, Std: {top_f1_diseases['hest_representation'].std():.2f}"
    )

    print(f"\nTop 15 ROCAUC improving diseases:")
    print(
        f"  H&E detectability - Mean: {top_rocauc_diseases['he_detectability'].mean():.2f}, Std: {top_rocauc_diseases['he_detectability'].std():.2f}"
    )
    print(
        f"  HEST representation - Mean: {top_rocauc_diseases['hest_representation'].mean():.2f}, Std: {top_rocauc_diseases['hest_representation'].std():.2f}"
    )

    print(f"\nAll diseases (n={len(analysis_df)}):")
    print(
        f"  H&E detectability - Mean: {analysis_df['he_detectability'].mean():.2f}, Std: {analysis_df['he_detectability'].std():.2f}"
    )
    print(
        f"  HEST representation - Mean: {analysis_df['hest_representation'].mean():.2f}, Std: {analysis_df['hest_representation'].std():.2f}"
    )

    plt.show()

    return top_f1_diseases, top_rocauc_diseases


def create_detectability_comparison_plots(analysis_df):
    """Create violin plots comparing low vs high H&E detectability diseases"""

    # Apply matplotlib style
    plt.style.use(snakemake.input.mpl_style)

    # Filter diseases by H&E detectability
    low_detectability = analysis_df[analysis_df["he_detectability"] <= 5].copy()
    high_detectability = analysis_df[analysis_df["he_detectability"] == 7].copy()

    print(f"\n=== DETECTABILITY COMPARISON ===")
    print(f"Low H&E detectability (≤5): {len(low_detectability)} diseases")
    print(f"High H&E detectability (=7): {len(high_detectability)} diseases")

    if len(low_detectability) == 0 or len(high_detectability) == 0:
        print("Not enough data for comparison plots")
        return

    # Prepare data for violin plots
    violin_data = []

    # Add low detectability data
    for _, row in low_detectability.iterrows():
        violin_data.append(
            {
                "group": f"Low H&E Detectability (≤5)\nn={len(low_detectability)}",
                "f1_improvement": row["f1_improvement"],
                "rocauc_improvement": row["rocauc_improvement"],
                "disease_name": row["disease_name"],
            }
        )

    # Add high detectability data
    for _, row in high_detectability.iterrows():
        violin_data.append(
            {
                "group": f"High H&E Detectability (=7)\nn={len(high_detectability)}",
                "f1_improvement": row["f1_improvement"],
                "rocauc_improvement": row["rocauc_improvement"],
                "disease_name": row["disease_name"],
            }
        )

    violin_df = pd.DataFrame(violin_data)

    # Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(6, 5))
    fig.suptitle("Performance Improvements: Low vs High H&E Detectability", fontsize=16)

    # F1 improvement plot
    sns.violinplot(
        data=violin_df, x="group", y="f1_improvement", ax=axes[0], inner="box"
    )
    sns.stripplot(
        data=violin_df,
        x="group",
        y="f1_improvement",
        ax=axes[0],
        size=4,
        alpha=0.7,
        color="black",
    )
    axes[0].set_title("F1 Score Improvement")
    axes[0].set_xlabel("H&E Detectability Group")
    axes[0].set_ylabel("F1 Improvement")
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis="x", rotation=45)
    for label in axes[0].get_xticklabels():
        label.set_ha("right")

    # ROCAUC improvement plot
    sns.violinplot(
        data=violin_df, x="group", y="rocauc_improvement", ax=axes[1], inner="box"
    )
    sns.stripplot(
        data=violin_df,
        x="group",
        y="rocauc_improvement",
        ax=axes[1],
        size=4,
        alpha=0.7,
        color="black",
    )
    axes[1].set_title("ROCAUC Improvement")
    axes[1].set_xlabel("H&E Detectability Group")
    axes[1].set_ylabel("ROCAUC Improvement")
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis="x", rotation=45)
    for label in axes[1].get_xticklabels():
        label.set_ha("right")

    plt.tight_layout()

    # Save the plot
    plt.savefig(
        snakemake.output.detectability_violin_plots, dpi=300, bbox_inches="tight"
    )
    print(
        f"\nDetectability comparison plots saved to: {snakemake.output.detectability_violin_plots}"
    )

    # Save as SVG
    plt.savefig(snakemake.output.detectability_violin_svg, bbox_inches="tight")
    print(f"SVG version saved to: {snakemake.output.detectability_violin_svg}")

    # Print summary statistics
    print(f"\n=== DETECTABILITY COMPARISON STATISTICS ===")

    print(f"\nLow H&E detectability (≤5) diseases:")
    print(
        f"  F1 improvement - Mean: {low_detectability['f1_improvement'].mean():.4f}, Std: {low_detectability['f1_improvement'].std():.4f}"
    )
    print(
        f"  ROCAUC improvement - Mean: {low_detectability['rocauc_improvement'].mean():.4f}, Std: {low_detectability['rocauc_improvement'].std():.4f}"
    )

    print(f"\nHigh H&E detectability (=7) diseases:")
    print(
        f"  F1 improvement - Mean: {high_detectability['f1_improvement'].mean():.4f}, Std: {high_detectability['f1_improvement'].std():.4f}"
    )
    print(
        f"  ROCAUC improvement - Mean: {high_detectability['rocauc_improvement'].mean():.4f}, Std: {high_detectability['rocauc_improvement'].std():.4f}"
    )

    # Show top diseases in each group
    print(f"\nTop 5 F1 improving diseases in low detectability group:")
    top_low_f1 = low_detectability.nlargest(5, "f1_improvement")
    for _, row in top_low_f1.iterrows():
        print(
            f"  {row['disease_name']}: F1={row['f1_improvement']:.4f}, H&E={row['he_detectability']}"
        )

    print(f"\nTop 5 F1 improving diseases in high detectability group:")
    top_high_f1 = high_detectability.nlargest(5, "f1_improvement")
    for _, row in top_high_f1.iterrows():
        print(
            f"  {row['disease_name']}: F1={row['f1_improvement']:.4f}, H&E={row['he_detectability']}"
        )

    plt.show()

    return violin_df


def main():
    """Main analysis function"""
    try:
        # Load data
        per_class_df, hest_repr_df = load_data()
        print(
            f"Loaded {len(per_class_df)} per-class entries and {len(hest_repr_df)} HEST representation entries"
        )

        # Prepare analysis data
        analysis_df = prepare_analysis_data(per_class_df, hest_repr_df)
        print(f"Created analysis dataset with {len(analysis_df)} diseases")

        if len(analysis_df) == 0:
            print("No overlapping diseases found between datasets!")
            return

        # Create distribution plots
        top_f1_diseases, top_rocauc_diseases = create_distribution_plots(analysis_df)

        # Create detectability comparison plots
        detectability_comparison_df = create_detectability_comparison_plots(analysis_df)

        # Save top diseases lists
        top_f1_diseases.to_csv(snakemake.output.top_f1_diseases, index=False)
        top_rocauc_diseases.to_csv(snakemake.output.top_rocauc_diseases, index=False)
        print(f"\nTop diseases lists saved to CSV files")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Please ensure per_class_analysis.csv and hest_representation_analysis.csv are in the current directory"
        )
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
