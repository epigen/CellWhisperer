#!/usr/bin/env python3
"""
Comprehensive analysis of high detectability diseases grouped by F1 performance changes
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re
from scipy import stats


def extract_disease_name(class_name):
    """Extract disease name from class description"""
    pattern = r"A sample of (.+?) from a healthy individual"
    match = re.search(pattern, class_name)
    if match:
        return match.group(1).lower().strip()
    return None


def load_original_evaluation_results():
    """Load original evaluation results with baseline scores and sample counts"""

    # Load trimodal results
    trimodal_path = Path(
        "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/spotwhisperer_eval/benchmarks/cellwhisperer/spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m/datasets/human_disease/celltype/performance_metrics_permetadataraw.csv"
    )

    if not trimodal_path.exists():
        raise FileNotFoundError(f"Trimodal results not found at {trimodal_path}")

    trimodal_df = pd.read_csv(trimodal_path)
    print(f"Loaded trimodal results: {len(trimodal_df)} classes")

    # Load bimodal matching results
    bimodal_path = Path(
        "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/spotwhisperer_eval/benchmarks/cellwhisperer/spotwhisperer_cellxgene_census__archs4_geo/datasets/human_disease/celltype/performance_metrics_permetadataraw.csv"
    )

    if not bimodal_path.exists():
        raise FileNotFoundError(f"Bimodal results not found at {bimodal_path}")

    bimodal_df = pd.read_csv(bimodal_path)
    print(f"Loaded bimodal results: {len(bimodal_df)} classes")

    return trimodal_df, bimodal_df


def load_detectability_scores():
    """Load H&E detectability scores from previous analysis"""

    detectability_df = pd.read_csv(snakemake.input.hest_representation_analysis)
    print(f"Loaded detectability scores for {len(detectability_df)} diseases")

    return detectability_df


def merge_and_analyze_data(trimodal_df, bimodal_df, detectability_df):
    """Merge all datasets and create comprehensive analysis"""

    # Merge trimodal and bimodal results on class
    merged_df = pd.merge(
        trimodal_df[
            [
                "class",
                "f1",
                "precision",
                "accuracy",
                "rocauc",
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "recall_at_50",
                "n_samples_in_class",
            ]
        ],
        bimodal_df[
            [
                "class",
                "f1",
                "precision",
                "accuracy",
                "rocauc",
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "recall_at_50",
            ]
        ],
        on="class",
        suffixes=("_trimodal", "_bimodal"),
    )

    print(f"Merged trimodal and bimodal results: {len(merged_df)} classes")

    # Calculate improvements for all metrics
    metrics = [
        "f1",
        "precision",
        "accuracy",
        "rocauc",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "recall_at_50",
    ]

    for metric in metrics:
        merged_df[f"{metric}_improvement"] = (
            merged_df[f"{metric}_trimodal"] - merged_df[f"{metric}_bimodal"]
        )
        merged_df[f"{metric}_relative_improvement"] = (
            merged_df[f"{metric}_improvement"] / merged_df[f"{metric}_bimodal"] * 100
        ).replace([np.inf, -np.inf], np.nan)

    # Extract disease names and merge with detectability scores
    merged_df["disease_name"] = merged_df["class"].apply(extract_disease_name)

    # Merge with detectability scores
    analysis_df = pd.merge(
        merged_df,
        detectability_df[["disease_name", "he_detectability"]],
        on="disease_name",
        how="inner",
    )

    print(
        f"Final analysis dataset: {len(analysis_df)} diseases with detectability scores"
    )

    return analysis_df


def categorize_high_detectability_diseases(analysis_df, f1_threshold=0.01):
    """Categorize high detectability diseases by F1 performance changes"""

    # Filter for high detectability diseases (score = 7)
    high_detectability = analysis_df[analysis_df["he_detectability"] == 7].copy()

    print(f"High detectability diseases (score=7): {len(high_detectability)}")

    # Categorize by F1 improvement
    def categorize_f1_change(f1_improvement):
        if f1_improvement > f1_threshold:
            return "Improving"
        elif f1_improvement < -f1_threshold:
            return "Deteriorating"
        else:
            return "Unchanged"

    high_detectability["f1_change_category"] = high_detectability[
        "f1_improvement"
    ].apply(categorize_f1_change)

    # Print category distribution
    category_counts = high_detectability["f1_change_category"].value_counts()
    print(f"\nF1 Change Categories (threshold = ±{f1_threshold}):")
    for category, count in category_counts.items():
        print(f"  {category}: {count} diseases")

    return high_detectability


def create_comprehensive_analysis(high_detectability_df):
    """Create comprehensive analysis of the three groups"""

    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE ANALYSIS OF HIGH DETECTABILITY DISEASES")
    print(f"{'='*60}")

    metrics = [
        "f1",
        "precision",
        "accuracy",
        "rocauc",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "recall_at_50",
    ]

    for category in ["Improving", "Deteriorating", "Unchanged"]:
        category_data = high_detectability_df[
            high_detectability_df["f1_change_category"] == category
        ]

        if len(category_data) == 0:
            continue

        print(f"\n{'-'*40}")
        print(f"{category.upper()} DISEASES (n={len(category_data)})")
        print(f"{'-'*40}")

        # Sample size statistics
        total_samples = category_data["n_samples_in_class"].sum()
        mean_samples = category_data["n_samples_in_class"].mean()
        median_samples = category_data["n_samples_in_class"].median()

        print(f"\nSample Size Statistics:")
        print(f"  Total samples across all diseases: {total_samples:,}")
        print(f"  Mean samples per disease: {mean_samples:.1f}")
        print(f"  Median samples per disease: {median_samples:.1f}")
        print(
            f"  Range: {category_data['n_samples_in_class'].min()} - {category_data['n_samples_in_class'].max()}"
        )

        # Baseline performance (bimodal scores)
        print(f"\nBaseline Performance (Bimodal Model):")
        for metric in metrics:
            baseline_col = f"{metric}_bimodal"
            if baseline_col in category_data.columns:
                values = category_data[baseline_col].dropna()
                if len(values) > 0:
                    print(
                        f"  {metric.upper()}: mean={values.mean():.4f}, median={values.median():.4f}, std={values.std():.4f}"
                    )

        # Final performance (trimodal scores)
        print(f"\nFinal Performance (Trimodal Model):")
        for metric in metrics:
            final_col = f"{metric}_trimodal"
            if final_col in category_data.columns:
                values = category_data[final_col].dropna()
                if len(values) > 0:
                    print(
                        f"  {metric.upper()}: mean={values.mean():.4f}, median={values.median():.4f}, std={values.std():.4f}"
                    )

        # Performance improvements
        print(f"\nPerformance Improvements:")
        for metric in metrics:
            improvement_col = f"{metric}_improvement"
            if improvement_col in category_data.columns:
                values = category_data[improvement_col].dropna()
                if len(values) > 0:
                    print(
                        f"  {metric.upper()}: mean={values.mean():.4f}, median={values.median():.4f}, std={values.std():.4f}"
                    )

        # List individual diseases
        print(f"\nIndividual Diseases:")
        category_sorted = category_data.sort_values("f1_improvement", ascending=False)
        for _, row in category_sorted.iterrows():
            print(
                f"  {row['disease_name']}: F1 {row['f1_improvement']:+.4f} ({row['n_samples_in_class']} samples, baseline F1={row['f1_bimodal']:.4f})"
            )


def create_visualization_plots(high_detectability_df, output_dir=Path(".")):
    """Create comprehensive visualization plots"""

    # Apply matplotlib style
    plt.style.use(snakemake.input.mpl_style)
    sns.set_palette("husl")

    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    fig.suptitle(
        "High Detectability Diseases: Comprehensive Analysis by F1 Change Category",
        fontsize=16,
    )

    categories = ["Improving", "Deteriorating", "Unchanged"]
    colors = ["green", "red", "gray"]

    # Plot 1: Sample size distribution
    ax = axes[0, 0]
    for i, category in enumerate(categories):
        category_data = high_detectability_df[
            high_detectability_df["f1_change_category"] == category
        ]
        if len(category_data) > 0:
            ax.hist(
                category_data["n_samples_in_class"],
                bins=20,
                alpha=0.6,
                label=f"{category} (n={len(category_data)})",
                color=colors[i],
            )
    ax.set_xlabel("Number of Samples per Disease")
    ax.set_ylabel("Frequency")
    ax.set_title("Sample Size Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Baseline F1 scores
    ax = axes[0, 1]
    baseline_data = []
    for category in categories:
        category_data = high_detectability_df[
            high_detectability_df["f1_change_category"] == category
        ]
        if len(category_data) > 0:
            baseline_data.append(category_data["f1_bimodal"].dropna())

    if baseline_data:
        ax.boxplot(baseline_data, labels=categories)
        ax.set_ylabel("Baseline F1 Score (Bimodal)")
        ax.set_title("Baseline F1 Score Distribution")
        ax.grid(True, alpha=0.3)

    # Plot 3: F1 improvements
    ax = axes[0, 2]
    improvement_data = []
    for category in categories:
        category_data = high_detectability_df[
            high_detectability_df["f1_change_category"] == category
        ]
        if len(category_data) > 0:
            improvement_data.append(category_data["f1_improvement"].dropna())

    if improvement_data:
        ax.boxplot(improvement_data, labels=categories)
        ax.set_ylabel("F1 Improvement")
        ax.set_title("F1 Improvement Distribution")
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax.grid(True, alpha=0.3)

    # Plot 4: Sample size vs F1 improvement scatter
    ax = axes[1, 0]
    for i, category in enumerate(categories):
        category_data = high_detectability_df[
            high_detectability_df["f1_change_category"] == category
        ]
        if len(category_data) > 0:
            ax.scatter(
                category_data["n_samples_in_class"],
                category_data["f1_improvement"],
                alpha=0.7,
                label=category,
                color=colors[i],
            )
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("F1 Improvement")
    ax.set_title("Sample Size vs F1 Improvement")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Baseline vs final F1 scores
    ax = axes[1, 1]
    texts = []
    for i, category in enumerate(categories):
        category_data = high_detectability_df[
            high_detectability_df["f1_change_category"] == category
        ]
        if len(category_data) > 0:
            scatter = ax.scatter(
                category_data["f1_bimodal"],
                category_data["f1_trimodal"],
                alpha=0.7,
                label=category,
                color=colors[i],
            )

            # Add text labels for disease names
            for _, row in category_data.iterrows():
                # Truncate long disease names for readability
                disease_name = row["disease_name"]
                if len(disease_name) > 20:
                    disease_name = disease_name[:17] + "..."

                texts.append(
                    ax.annotate(
                        disease_name,
                        (row["f1_bimodal"], row["f1_trimodal"]),
                        fontsize=8,
                        alpha=0.8,
                    )
                )

    # Add diagonal line (no improvement)
    min_val = min(
        high_detectability_df["f1_bimodal"].min(),
        high_detectability_df["f1_trimodal"].min(),
    )
    max_val = max(
        high_detectability_df["f1_bimodal"].max(),
        high_detectability_df["f1_trimodal"].max(),
    )
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.7, label="No change")

    # Adjust text positions to avoid overlaps
    try:
        from adjustText import adjust_text

        adjust_text(
            texts, ax=ax, arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5)
        )
    except ImportError:
        print("Warning: adjustText not available, labels may overlap")

    ax.set_xlabel("Baseline F1 (Bimodal)")
    ax.set_ylabel("Final F1 (Trimodal)")
    ax.set_title("Baseline vs Final F1 Scores")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: ROCAUC improvements
    ax = axes[1, 2]
    rocauc_data = []
    for category in categories:
        category_data = high_detectability_df[
            high_detectability_df["f1_change_category"] == category
        ]
        if len(category_data) > 0:
            rocauc_data.append(category_data["rocauc_improvement"].dropna())

    if rocauc_data:
        ax.boxplot(rocauc_data, labels=categories)
        ax.set_ylabel("ROCAUC Improvement")
        ax.set_title("ROCAUC Improvement Distribution")
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax.grid(True, alpha=0.3)

    # Plot 7: Correlation matrix for improving diseases
    ax = axes[2, 0]
    improving_data = high_detectability_df[
        high_detectability_df["f1_change_category"] == "Improving"
    ]
    if len(improving_data) > 1:
        corr_cols = [
            "n_samples_in_class",
            "f1_bimodal",
            "f1_improvement",
            "rocauc_improvement",
        ]
        corr_data = improving_data[corr_cols].corr()
        sns.heatmap(corr_data, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Matrix (Improving Diseases)")

    # Plot 8: Performance across all metrics for each category
    ax = axes[2, 1]
    metrics = ["f1", "precision", "accuracy", "rocauc"]
    category_means = {}

    for category in categories:
        category_data = high_detectability_df[
            high_detectability_df["f1_change_category"] == category
        ]
        if len(category_data) > 0:
            means = [
                category_data[f"{metric}_improvement"].mean() for metric in metrics
            ]
            category_means[category] = means

    x = np.arange(len(metrics))
    width = 0.25

    for i, (category, means) in enumerate(category_means.items()):
        ax.bar(x + i * width, means, width, label=category, color=colors[i], alpha=0.7)

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Mean Improvement")
    ax.set_title("Mean Improvement Across Metrics")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 9: Statistical significance tests
    ax = axes[2, 2]

    # Perform statistical tests between groups
    improving = high_detectability_df[
        high_detectability_df["f1_change_category"] == "Improving"
    ]["f1_improvement"]
    deteriorating = high_detectability_df[
        high_detectability_df["f1_change_category"] == "Deteriorating"
    ]["f1_improvement"]
    unchanged = high_detectability_df[
        high_detectability_df["f1_change_category"] == "Unchanged"
    ]["f1_improvement"]

    # T-tests
    test_results = []
    if len(improving) > 1 and len(deteriorating) > 1:
        t_stat, p_val = stats.ttest_ind(improving, deteriorating)
        test_results.append(
            f"Improving vs Deteriorating:\nt={t_stat:.3f}, p={p_val:.4f}"
        )

    if len(improving) > 1 and len(unchanged) > 1:
        t_stat, p_val = stats.ttest_ind(improving, unchanged)
        test_results.append(f"Improving vs Unchanged:\nt={t_stat:.3f}, p={p_val:.4f}")

    if len(deteriorating) > 1 and len(unchanged) > 1:
        t_stat, p_val = stats.ttest_ind(deteriorating, unchanged)
        test_results.append(
            f"Deteriorating vs Unchanged:\nt={t_stat:.3f}, p={p_val:.4f}"
        )

    ax.text(
        0.1,
        0.9,
        "\n\n".join(test_results),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_title("Statistical Tests (F1 Improvements)")
    ax.axis("off")

    plt.tight_layout()

    # Save plot
    plt.savefig(snakemake.output.comprehensive_plots, dpi=300, bbox_inches="tight")
    print(
        f"\nComprehensive analysis plots saved to: {snakemake.output.comprehensive_plots}"
    )

    # Save as SVG
    plt.savefig(snakemake.output.comprehensive_plots_svg, bbox_inches="tight")
    print(f"SVG version saved to: {snakemake.output.comprehensive_plots_svg}")

    plt.close()


def save_detailed_results(high_detectability_df, output_dir=Path(".")):
    """Save detailed results to CSV files"""

    # Save full analysis dataset
    high_detectability_df.to_csv(snakemake.output.full_analysis, index=False)
    print(f"Full analysis data saved to: {snakemake.output.full_analysis}")

    # Save summary by category
    summary_data = []

    for category in ["Improving", "Deteriorating", "Unchanged"]:
        category_data = high_detectability_df[
            high_detectability_df["f1_change_category"] == category
        ]

        if len(category_data) == 0:
            continue

        summary_data.append(
            {
                "category": category,
                "n_diseases": len(category_data),
                "total_samples": category_data["n_samples_in_class"].sum(),
                "mean_samples_per_disease": category_data["n_samples_in_class"].mean(),
                "median_samples_per_disease": category_data[
                    "n_samples_in_class"
                ].median(),
                "mean_baseline_f1": category_data["f1_bimodal"].mean(),
                "mean_final_f1": category_data["f1_trimodal"].mean(),
                "mean_f1_improvement": category_data["f1_improvement"].mean(),
                "std_f1_improvement": category_data["f1_improvement"].std(),
                "mean_rocauc_improvement": category_data["rocauc_improvement"].mean(),
                "mean_precision_improvement": category_data[
                    "precision_improvement"
                ].mean(),
                "mean_accuracy_improvement": category_data[
                    "accuracy_improvement"
                ].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(snakemake.output.category_summary, index=False)
    print(f"Category summary saved to: {snakemake.output.category_summary}")


def main():
    """Main analysis function"""
    try:
        # Load all required data
        print("Loading original evaluation results...")
        trimodal_df, bimodal_df = load_original_evaluation_results()

        print("Loading detectability scores...")
        detectability_df = load_detectability_scores()

        # Merge and analyze
        print("Merging datasets and calculating improvements...")
        analysis_df = merge_and_analyze_data(trimodal_df, bimodal_df, detectability_df)

        # Categorize high detectability diseases
        print("Categorizing high detectability diseases...")
        high_detectability_df = categorize_high_detectability_diseases(analysis_df)

        # Create comprehensive analysis
        create_comprehensive_analysis(high_detectability_df)

        # Create visualizations
        print("\nCreating visualization plots...")
        create_visualization_plots(high_detectability_df)

        # Save detailed results
        print("\nSaving detailed results...")
        save_detailed_results(high_detectability_df)

        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all required files are available")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
