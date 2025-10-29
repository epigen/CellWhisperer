#!/usr/bin/env python3
"""
Adhoc analysis of per-class results for human_disease dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re
from collections import Counter


def extract_disease_name(class_name):
    """Extract disease name from class description"""
    # Pattern to match "A sample of [disease] from a healthy individual"
    pattern = r"A sample of (.+?) from a healthy individual"
    match = re.search(pattern, class_name)
    if match:
        return match.group(1).lower().strip()
    return None


def count_disease_mentions(disease_name, quilt_df):
    """Count mentions of disease in quilt1m captions"""
    if not disease_name:
        return 0

    # Count mentions in caption column (case-insensitive)
    count = 0
    for caption in quilt_df["caption"].fillna(""):
        if disease_name in caption.lower():
            count += 1
    return count


def main():
    # Load the CSV file from snakemake input
    df = pd.read_csv(snakemake.input.per_class_analysis)

    # Load quilt1m lookup data
    quilt_path = Path(
        "~/cellwhisperer_private/results/quilt1m/quilt_1M_lookup.csv"
    ).expanduser()
    try:
        quilt_df = pd.read_csv(quilt_path)
        print(f"Loaded quilt1m data with {len(quilt_df)} entries")
    except FileNotFoundError:
        print(f"Warning: Could not find quilt1m data at {quilt_path}")
        print("Skipping disease mention correlation analysis")
        quilt_df = None

    # Filter for human_disease dataset
    human_disease_df = df[df["dataset"] == "human_disease"].copy()

    print(f"Found {len(human_disease_df)} human_disease entries")

    # Check what columns are available
    print("Available columns:", human_disease_df.columns.tolist())
    print("\nSample of data:")
    print(human_disease_df.head())

    # Check unique metrics available
    print(f"\nUnique metrics: {human_disease_df['metric'].unique()}")

    # Filter for rocauc and f1 metrics separately
    rocauc_data = human_disease_df[human_disease_df["metric"] == "rocauc"].copy()
    f1_data = human_disease_df[human_disease_df["metric"] == "f1"].copy()

    print(f"\nROCAUC metric entries: {len(rocauc_data)}")
    print(f"F1 metric entries: {len(f1_data)}")

    # Remove rows with NaN improvements for each metric
    rocauc_data = rocauc_data.dropna(subset=["rocauc_improvement"])
    f1_data = f1_data.dropna(subset=["f1_improvement"])

    print(f"ROCAUC entries after removing NaN: {len(rocauc_data)}")
    print(f"F1 entries after removing NaN: {len(f1_data)}")

    # Check if we have any data left
    if len(rocauc_data) == 0:
        print("\nNo valid ROCAUC data found for human_disease dataset!")
        print(f"Available datasets in CSV: {df['dataset'].unique()}")
        return

    if len(f1_data) == 0:
        print("\nNo valid F1 data found for human_disease dataset!")
        return

    # Merge rocauc and f1 data on class
    merged_data = pd.merge(
        rocauc_data[["class", "rocauc_improvement"]],
        f1_data[["class", "f1_improvement"]],
        on="class",
        how="inner",
    )

    print(f"Merged data entries: {len(merged_data)}")

    if len(merged_data) == 0:
        print("\nNo overlapping classes between ROCAUC and F1 data!")
        return

    # Get top 15 samples with strongest rocauc improvement
    top_15_rocauc = merged_data.nlargest(15, "rocauc_improvement")

    print("\nTop 15 classes with strongest ROCAUC improvement:")
    for _, row in top_15_rocauc.iterrows():
        print(f"  {row['class']}: {row['rocauc_improvement']:.4f}")

    # Get top 15 samples with strongest f1 improvement
    top_15_f1 = merged_data.nlargest(15, "f1_improvement")

    print("\nTop 15 classes with strongest F1 improvement:")
    for _, row in top_15_f1.iterrows():
        print(f"  {row['class']}: {row['f1_improvement']:.4f}")

    # Get diseases where both F1 and ROCAUC improvements are > 0.1
    both_improved = merged_data[
        (merged_data["f1_improvement"] > 0.1)
        & (merged_data["rocauc_improvement"] > 0.1)
    ].copy()

    print(
        f"\nDiseases with both F1 and ROCAUC improvements > 0.1 ({len(both_improved)} total):"
    )
    # Sort by combined improvement score for ranking
    both_improved["combined_improvement"] = (
        both_improved["f1_improvement"] + both_improved["rocauc_improvement"]
    )
    both_improved_sorted = both_improved.nlargest(15, "combined_improvement")

    for _, row in both_improved_sorted.iterrows():
        print(
            f"  {row['class']}: F1={row['f1_improvement']:.4f}, ROCAUC={row['rocauc_improvement']:.4f}"
        )

    # Apply matplotlib style
    plt.style.use(snakemake.input.mpl_style)

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Distribution of f1 improvement for top 15 rocauc improving samples
    ax1.hist(
        top_15_rocauc["f1_improvement"],
        bins=10,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    ax1.set_xlabel("F1 Improvement")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of F1 Improvement\n(Top 15 ROCAUC Improving Classes)")
    ax1.grid(True, alpha=0.3)

    # Add statistics to the plot
    mean_f1 = top_15_rocauc["f1_improvement"].mean()
    median_f1 = top_15_rocauc["f1_improvement"].median()
    ax1.axvline(mean_f1, color="red", linestyle="--", label=f"Mean: {mean_f1:.4f}")
    ax1.axvline(
        median_f1, color="orange", linestyle="--", label=f"Median: {median_f1:.4f}"
    )
    ax1.legend()

    # Plot 2: Correlation between f1 improvement and rocauc improvement for all human_disease
    ax2.scatter(
        merged_data["rocauc_improvement"],
        merged_data["f1_improvement"],
        alpha=0.6,
        color="green",
    )

    # Calculate and display correlation
    correlation = merged_data["rocauc_improvement"].corr(merged_data["f1_improvement"])

    # Add trend line
    z = np.polyfit(merged_data["rocauc_improvement"], merged_data["f1_improvement"], 1)
    p = np.poly1d(z)
    ax2.plot(
        merged_data["rocauc_improvement"],
        p(merged_data["rocauc_improvement"]),
        "r--",
        alpha=0.8,
        label=f"Trend line (r={correlation:.3f})",
    )

    ax2.set_xlabel("ROCAUC Improvement")
    ax2.set_ylabel("F1 Improvement")
    ax2.set_title("Correlation: F1 vs ROCAUC Improvement\n(All Human Disease Classes)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save the plot
    plt.savefig(snakemake.output.analysis_plots, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {snakemake.output.analysis_plots}")

    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Human disease dataset entries: {len(merged_data)}")
    print(f"Top 15 ROCAUC improving classes:")
    print(f"  F1 improvement - Mean: {mean_f1:.4f}, Median: {median_f1:.4f}")
    print(f"  F1 improvement - Min: {top_15_rocauc['f1_improvement'].min():.4f}")
    print(f"  F1 improvement - Max: {top_15_rocauc['f1_improvement'].max():.4f}")
    print(f"\nCorrelation between ROCAUC and F1 improvement: {correlation:.4f}")

    # Disease mention correlation analysis
    if quilt_df is not None:
        print("\n=== DISEASE MENTION CORRELATION ANALYSIS ===")

        # Extract disease names and count mentions
        disease_data = []
        for _, row in merged_data.iterrows():
            disease_name = extract_disease_name(row["class"])
            if disease_name:
                mention_count = count_disease_mentions(disease_name, quilt_df)
                disease_data.append(
                    {
                        "class": row["class"],
                        "disease_name": disease_name,
                        "mention_count": mention_count,
                        "log_mention_count": np.log1p(
                            mention_count
                        ),  # log(1 + count) to handle 0 counts
                        "f1_improvement": row["f1_improvement"],
                        "rocauc_improvement": row["rocauc_improvement"],
                        "f1_relative_improvement": (
                            row["f1_improvement"]
                            / merged_data[merged_data["class"] == row["class"]][
                                "f1_improvement"
                            ].iloc[0]
                            * 100
                            if row["f1_improvement"] != 0
                            else 0
                        ),
                        "rocauc_relative_improvement": (
                            row["rocauc_improvement"]
                            / merged_data[merged_data["class"] == row["class"]][
                                "rocauc_improvement"
                            ].iloc[0]
                            * 100
                            if row["rocauc_improvement"] != 0
                            else 0
                        ),
                    }
                )

        disease_df = pd.DataFrame(disease_data)
        print(f"Extracted {len(disease_df)} diseases with mention counts")

        # Use all diseases for log-count analysis (log1p handles 0 counts)
        disease_df_filtered = disease_df.copy()
        print(f"Total diseases for log-count analysis: {len(disease_df_filtered)}")

        if len(disease_df_filtered) > 5:  # Only create plots if we have enough data
            # Apply matplotlib style
            plt.style.use(snakemake.input.mpl_style)

            # Create correlation plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(
                "Disease Performance vs Quilt1M Mention Frequency", fontsize=16
            )

            metrics = [
                ("f1_improvement", "F1 Improvement"),
                ("rocauc_improvement", "ROCAUC Improvement"),
                ("f1_relative_improvement", "F1 Relative Improvement (%)"),
                ("rocauc_relative_improvement", "ROCAUC Relative Improvement (%)"),
            ]

            for i, (metric, title) in enumerate(metrics):
                ax = axes[i // 2, i % 2]

                # Scatter plot using log counts
                ax.scatter(
                    disease_df_filtered["log_mention_count"],
                    disease_df_filtered[metric],
                    alpha=0.6,
                    color="blue",
                )

                # Calculate correlation with log counts
                corr = disease_df_filtered["log_mention_count"].corr(
                    disease_df_filtered[metric]
                )

                # Add trend line if correlation is meaningful
                if not np.isnan(corr) and len(disease_df_filtered) > 2:
                    z = np.polyfit(
                        disease_df_filtered["log_mention_count"],
                        disease_df_filtered[metric],
                        1,
                    )
                    p = np.poly1d(z)
                    ax.plot(
                        disease_df_filtered["log_mention_count"],
                        p(disease_df_filtered["log_mention_count"]),
                        "r--",
                        alpha=0.8,
                        label=f"r={corr:.3f}",
                    )
                    ax.legend()

                ax.set_xlabel("Log(1 + Mentions) in Quilt1M Dataset")
                ax.set_ylabel(title)
                ax.set_title(f"{title} vs Mention Count")
                ax.grid(True, alpha=0.3)

                # Add some top disease labels
                top_diseases = disease_df_filtered.nlargest(3, metric)
                for _, disease_row in top_diseases.iterrows():
                    ax.annotate(
                        disease_row["disease_name"][:20] + "...",
                        (disease_row["log_mention_count"], disease_row[metric]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                    )

            plt.tight_layout()

            # Save the disease correlation plot
            plt.savefig(
                snakemake.output.mention_correlation_plots, dpi=300, bbox_inches="tight"
            )
            print(
                f"\nDisease mention correlation plot saved to: {snakemake.output.mention_correlation_plots}"
            )

            # Print correlation statistics
            print(f"\nDisease mention correlations (using log counts):")
            for metric, title in metrics:
                corr = disease_df_filtered["log_mention_count"].corr(
                    disease_df_filtered[metric]
                )
                print(f"  {title}: r={corr:.4f}")

            # Show top mentioned diseases
            print(f"\nTop 10 most mentioned diseases in Quilt1M:")
            top_mentioned = disease_df.nlargest(10, "mention_count")
            for _, row in top_mentioned.iterrows():
                print(f"  {row['disease_name']}: {row['mention_count']} mentions")

            # Apply matplotlib style
            plt.style.use(snakemake.input.mpl_style)

            # Create additional violin plot with grouped counts - separate axes for each metric
            fig_violin, axes_violin = plt.subplots(2, 2, figsize=(15, 12))
            fig_violin.suptitle(
                "Distribution of Performance Improvements by Mention Count Groups",
                fontsize=16,
            )

            # Create count groups
            disease_df_filtered["count_group"] = disease_df_filtered[
                "mention_count"
            ].apply(lambda x: "Counts ≥ 10" if x >= 10 else "Counts < 10")

            if len(disease_df_filtered) > 0:
                import seaborn as sns

                for i, (metric, title) in enumerate(metrics):
                    ax = axes_violin[i // 2, i % 2]

                    # Prepare data for this specific metric
                    metric_violin_data = []
                    for count_group in ["Counts < 10", "Counts ≥ 10"]:
                        group_data = disease_df_filtered[
                            disease_df_filtered["count_group"] == count_group
                        ]
                        for value in group_data[metric]:
                            metric_violin_data.append(
                                {
                                    "count_group": count_group,
                                    "improvement": value,
                                }
                            )

                    metric_violin_df = pd.DataFrame(metric_violin_data)

                    if len(metric_violin_df) > 0:
                        # Create violin plot for this metric
                        sns.violinplot(
                            data=metric_violin_df,
                            x="count_group",
                            y="improvement",
                            inner="box",
                            ax=ax,
                        )

                        ax.set_xlabel("Mention Count Groups")
                        ax.set_ylabel("Improvement Score")
                        ax.set_title(f"{title}")
                        ax.grid(True, alpha=0.3)

                        # Add sample size annotations
                        low_count = len(
                            disease_df_filtered[
                                (disease_df_filtered["count_group"] == "Counts < 10")
                            ]
                        )
                        high_count = len(
                            disease_df_filtered[
                                (disease_df_filtered["count_group"] == "Counts ≥ 10")
                            ]
                        )

                        # Add text at top of plot
                        y_max = ax.get_ylim()[1]
                        ax.text(
                            0,
                            y_max * 0.95,
                            f"n={low_count}",
                            ha="center",
                            va="top",
                            fontsize=10,
                            alpha=0.8,
                        )
                        ax.text(
                            1,
                            y_max * 0.95,
                            f"n={high_count}",
                            ha="center",
                            va="top",
                            fontsize=10,
                            alpha=0.8,
                        )

                plt.tight_layout()

                # Save the violin plot
                plt.savefig(snakemake.output.violin_plots, dpi=300, bbox_inches="tight")
                print(f"\nViolin plot saved to: {snakemake.output.violin_plots}")

                # Print statistics for each group
                print(f"\n=== COUNT GROUP STATISTICS ===")
                for count_group in ["Counts < 10", "Counts ≥ 10"]:
                    group_data = disease_df_filtered[
                        disease_df_filtered["count_group"] == count_group
                    ]
                    print(f"\n{count_group} (n={len(group_data)}):")
                    for metric, title in metrics:
                        values = group_data[metric]
                        print(
                            f"  {title}: mean={values.mean():.4f}, median={values.median():.4f}, std={values.std():.4f}"
                        )

                plt.show()
            else:
                print("No data available for violin plot")

            plt.show()
        else:
            print(
                "Not enough diseases with mentions to create meaningful correlation plots"
            )

    plt.show()


if __name__ == "__main__":
    main()
