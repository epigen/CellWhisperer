#!/usr/bin/env python3
"""
Analysis of correlation between absolute F1 score changes and Quilt1M disease mentions
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re
from scipy.stats import pearsonr, spearmanr


def extract_disease_name(class_name):
    """Extract disease name from class description"""
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


def load_and_prepare_data():
    """Load per-class analysis data and quilt1m lookup data"""

    # Load per-class analysis data
    df = pd.read_csv(snakemake.input.per_class_analysis)

    # Load quilt1m lookup data
    quilt_path = Path(
        "~/cellwhisperer_private/results/quilt1m/quilt_1M_lookup.csv"
    ).expanduser()
    if not quilt_path.exists():
        raise FileNotFoundError(f"Quilt1M data not found at {quilt_path}")

    quilt_df = pd.read_csv(quilt_path)
    print(f"Loaded quilt1m data with {len(quilt_df)} entries")

    # Filter for human_disease dataset
    human_disease_df = df[df["dataset"] == "human_disease"].copy()

    # Filter for F1 metric
    f1_data = human_disease_df[human_disease_df["metric"] == "f1"].copy()
    f1_data = f1_data.dropna(subset=["f1_improvement"])

    print(f"Found {len(f1_data)} human disease F1 entries")

    return f1_data, quilt_df


def analyze_mention_correlation(f1_data, quilt_df):
    """Analyze correlation between F1 improvements and Quilt1M mentions"""

    # Extract disease names and count mentions
    analysis_data = []

    for _, row in f1_data.iterrows():
        disease_name = extract_disease_name(row["class"])
        if disease_name:
            mention_count = count_disease_mentions(disease_name, quilt_df)

            analysis_data.append(
                {
                    "class": row["class"],
                    "disease_name": disease_name,
                    "f1_improvement": row["f1_improvement"],
                    "f1_absolute_change": abs(row["f1_improvement"]),  # Absolute change
                    "mention_count": mention_count,
                    "log_mention_count": np.log1p(mention_count),  # log(1 + count)
                    "has_mentions": mention_count > 0,
                }
            )

    analysis_df = pd.DataFrame(analysis_data)
    print(f"Created analysis dataset with {len(analysis_df)} diseases")

    return analysis_df


def create_correlation_plots(analysis_df):
    """Create correlation plots for absolute F1 changes vs mentions"""

    # Apply matplotlib style
    plt.style.use(snakemake.input.mpl_style)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Absolute F1 Score Changes vs Quilt1M Disease Mentions", fontsize=16)

    # Plot 1: Absolute F1 change vs raw mention count
    ax = axes[0, 0]
    ax.scatter(
        analysis_df["mention_count"], analysis_df["f1_absolute_change"], alpha=0.6
    )

    # Calculate correlation
    corr_raw, p_val_raw = pearsonr(
        analysis_df["mention_count"], analysis_df["f1_absolute_change"]
    )
    spearman_raw, sp_p_raw = spearmanr(
        analysis_df["mention_count"], analysis_df["f1_absolute_change"]
    )

    ax.set_xlabel("Mention Count in Quilt1M")
    ax.set_ylabel("Absolute F1 Score Change")
    ax.set_title(
        f"Raw Mention Count\nPearson r={corr_raw:.3f} (p={p_val_raw:.3f})\nSpearman ρ={spearman_raw:.3f}"
    )
    ax.grid(True, alpha=0.3)

    # Plot 2: Absolute F1 change vs log mention count
    ax = axes[0, 1]
    ax.scatter(
        analysis_df["log_mention_count"], analysis_df["f1_absolute_change"], alpha=0.6
    )

    # Calculate correlation with log counts
    corr_log, p_val_log = pearsonr(
        analysis_df["log_mention_count"], analysis_df["f1_absolute_change"]
    )
    spearman_log, sp_p_log = spearmanr(
        analysis_df["log_mention_count"], analysis_df["f1_absolute_change"]
    )

    ax.set_xlabel("Log(1 + Mention Count)")
    ax.set_ylabel("Absolute F1 Score Change")
    ax.set_title(
        f"Log Mention Count\nPearson r={corr_log:.3f} (p={p_val_log:.3f})\nSpearman ρ={spearman_log:.3f}"
    )
    ax.grid(True, alpha=0.3)

    # Plot 3: Box plot comparing diseases with/without mentions
    ax = axes[1, 0]
    mention_groups = analysis_df.groupby("has_mentions")["f1_absolute_change"]

    box_data = [mention_groups.get_group(False), mention_groups.get_group(True)]
    box_labels = ["No Mentions", "Has Mentions"]

    ax.boxplot(box_data, labels=box_labels)
    ax.set_ylabel("Absolute F1 Score Change")
    ax.set_title("Absolute F1 Changes: Mentioned vs Not Mentioned")
    ax.grid(True, alpha=0.3)

    # Add sample sizes
    no_mentions_count = (~analysis_df["has_mentions"]).sum()
    has_mentions_count = analysis_df["has_mentions"].sum()
    ax.text(1, ax.get_ylim()[1] * 0.9, f"n={no_mentions_count}", ha="center")
    ax.text(2, ax.get_ylim()[1] * 0.9, f"n={has_mentions_count}", ha="center")

    # Plot 4: Histogram of mention counts
    ax = axes[1, 1]
    ax.hist(analysis_df["mention_count"], bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Mention Count")
    ax.set_ylabel("Number of Diseases")
    ax.set_title("Distribution of Mention Counts")
    ax.grid(True, alpha=0.3)

    # Add statistics text
    ax.text(
        0.7,
        0.8,
        f"Mean: {analysis_df['mention_count'].mean():.1f}\n"
        f"Median: {analysis_df['mention_count'].median():.1f}\n"
        f"Max: {analysis_df['mention_count'].max()}\n"
        f"Zero mentions: {(analysis_df['mention_count'] == 0).sum()}",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save plot
    plt.savefig(snakemake.output.correlation_plots, dpi=300, bbox_inches="tight")
    print(f"Correlation plots saved to: {snakemake.output.correlation_plots}")

    plt.show()

    return {
        "pearson_raw": (corr_raw, p_val_raw),
        "spearman_raw": (spearman_raw, sp_p_raw),
        "pearson_log": (corr_log, p_val_log),
        "spearman_log": (spearman_log, sp_p_log),
    }


def print_detailed_analysis(analysis_df, correlations):
    """Print detailed statistical analysis"""

    print(f"\n=== QUILT1M MENTION CORRELATION ANALYSIS ===")
    print(f"Total diseases analyzed: {len(analysis_df)}")
    print(f"Diseases with mentions: {analysis_df['has_mentions'].sum()}")
    print(f"Diseases without mentions: {(~analysis_df['has_mentions']).sum()}")

    print(f"\n=== CORRELATION RESULTS ===")
    print(f"Raw mention count vs absolute F1 change:")
    print(
        f"  Pearson correlation: r={correlations['pearson_raw'][0]:.4f}, p={correlations['pearson_raw'][1]:.4f}"
    )
    print(
        f"  Spearman correlation: ρ={correlations['spearman_raw'][0]:.4f}, p={correlations['spearman_raw'][1]:.4f}"
    )

    print(f"\nLog mention count vs absolute F1 change:")
    print(
        f"  Pearson correlation: r={correlations['pearson_log'][0]:.4f}, p={correlations['pearson_log'][1]:.4f}"
    )
    print(
        f"  Spearman correlation: ρ={correlations['spearman_log'][0]:.4f}, p={correlations['spearman_log'][1]:.4f}"
    )

    # Group comparison
    mentioned_diseases = analysis_df[analysis_df["has_mentions"]]
    not_mentioned_diseases = analysis_df[~analysis_df["has_mentions"]]

    print(f"\n=== GROUP COMPARISON ===")
    print(f"Diseases with mentions (n={len(mentioned_diseases)}):")
    print(
        f"  Mean absolute F1 change: {mentioned_diseases['f1_absolute_change'].mean():.4f}"
    )
    print(
        f"  Median absolute F1 change: {mentioned_diseases['f1_absolute_change'].median():.4f}"
    )
    print(
        f"  Std absolute F1 change: {mentioned_diseases['f1_absolute_change'].std():.4f}"
    )

    print(f"\nDiseases without mentions (n={len(not_mentioned_diseases)}):")
    print(
        f"  Mean absolute F1 change: {not_mentioned_diseases['f1_absolute_change'].mean():.4f}"
    )
    print(
        f"  Median absolute F1 change: {not_mentioned_diseases['f1_absolute_change'].median():.4f}"
    )
    print(
        f"  Std absolute F1 change: {not_mentioned_diseases['f1_absolute_change'].std():.4f}"
    )

    # Statistical test
    from scipy.stats import mannwhitneyu

    statistic, p_value = mannwhitneyu(
        mentioned_diseases["f1_absolute_change"],
        not_mentioned_diseases["f1_absolute_change"],
        alternative="two-sided",
    )
    print(f"\nMann-Whitney U test (mentioned vs not mentioned):")
    print(f"  U-statistic: {statistic:.2f}")
    print(f"  p-value: {p_value:.4f}")

    # Top mentioned diseases
    print(f"\n=== TOP MENTIONED DISEASES ===")
    top_mentioned = analysis_df.nlargest(10, "mention_count")
    for _, row in top_mentioned.iterrows():
        print(
            f"  {row['disease_name']}: {row['mention_count']} mentions, "
            f"abs F1 change: {row['f1_absolute_change']:.4f}"
        )

    # Diseases with highest absolute F1 changes
    print(f"\n=== DISEASES WITH HIGHEST ABSOLUTE F1 CHANGES ===")
    top_f1_changes = analysis_df.nlargest(10, "f1_absolute_change")
    for _, row in top_f1_changes.iterrows():
        print(
            f"  {row['disease_name']}: abs F1 change: {row['f1_absolute_change']:.4f}, "
            f"{row['mention_count']} mentions"
        )


def main():
    """Main analysis function"""
    try:
        # Load data
        f1_data, quilt_df = load_and_prepare_data()

        # Analyze correlation
        analysis_df = analyze_mention_correlation(f1_data, quilt_df)

        if len(analysis_df) == 0:
            print("No diseases found for analysis!")
            return

        # Create plots and get correlation results
        correlations = create_correlation_plots(analysis_df)

        # Print detailed analysis
        print_detailed_analysis(analysis_df, correlations)

        # Save analysis data
        analysis_df.to_csv(snakemake.output.correlation_data, index=False)
        print(f"\nAnalysis data saved to: {snakemake.output.correlation_data}")

        # Answer the main question
        print(f"\n=== ANSWER TO YOUR QUESTION ===")
        pearson_r = correlations["pearson_log"][0]
        pearson_p = correlations["pearson_log"][1]

        if pearson_p < 0.05:
            significance = "statistically significant"
        else:
            significance = "not statistically significant"

        print(
            f"Does the absolute change in F1 score correlate with Quilt1M disease mentions?"
        )
        print(
            f"Answer: The correlation is {pearson_r:.4f} (using log-transformed mention counts), "
        )
        print(f"which is {significance} (p={pearson_p:.4f}).")

        if abs(pearson_r) < 0.1:
            strength = "very weak"
        elif abs(pearson_r) < 0.3:
            strength = "weak"
        elif abs(pearson_r) < 0.5:
            strength = "moderate"
        else:
            strength = "strong"

        print(f"This represents a {strength} correlation.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Please ensure per_class_analysis.csv exists and quilt1m data is available"
        )
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
