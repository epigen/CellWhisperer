#!/usr/bin/env python3
"""
LLM-based analysis of disease detectability through H&E histopathology
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import openai
import time
import os


def extract_disease_name(class_name: str) -> Optional[str]:
    """Extract disease name from class description"""
    # Pattern to match "A sample of [disease] from a healthy individual"
    pattern = r"A sample of (.+?) from a healthy individual"
    match = re.search(pattern, class_name)
    if match:
        return match.group(1).lower().strip()
    return None


def query_llm_for_histopathology_detectability(disease_name: str, client) -> int:
    """
    Query LLM to determine if a disease is detectable through H&E histopathology

    Returns integer score from 1-7 (1=very unlikely, 7=very likely)
    """
    prompt = f"""
Rate how likely the disease "{disease_name}" is to be detectable or diagnosable through H&E (hematoxylin and eosin) histopathology analysis.

Consider:
- Whether the disease causes visible morphological changes in tissue
- Whether H&E staining would reveal characteristic features
- Whether pathologists routinely use H&E to diagnose this condition
- Whether the disease affects tissue architecture, cell morphology, or cellular organization

Rate on a scale from 1 to 7:
1 = Very unlikely to be detectable through H&E
2 = Unlikely to be detectable through H&E
3 = Somewhat unlikely to be detectable through H&E
4 = Neutral/uncertain
5 = Somewhat likely to be detectable through H&E
6 = Likely to be detectable through H&E
7 = Very likely to be detectable through H&E

Answer with only the number (1-7).
"""

    try:
        response = client.chat.completions.create(
            model="claude-4-sonnet",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical expert specializing in histopathology. Provide concise, accurate numerical ratings about disease detectability through H&E staining.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
            temperature=0.1,
        )

        answer = response.choices[0].message.content.strip()
        try:
            score = int(answer)
            if 1 <= score <= 7:
                return score
            else:
                print(f"Invalid score {score} for {disease_name}, skipping")
                return None
        except ValueError:
            print(f"Could not parse score '{answer}' for {disease_name}, skipping")
            return None

    except Exception as e:
        print(f"Error querying LLM for {disease_name}: {e}")
        return None


def classify_diseases_by_histopathology(
    disease_names: List[str], output_path: Path
) -> Dict[str, int]:
    """
    Classify diseases by their detectability through H&E histopathology using LLM

    Returns dictionary mapping disease name to detectability score (1-7)
    """
    # Check if results already exist
    if output_path.exists():
        print(f"Loading existing classification from {output_path}")
        with open(output_path, "r") as f:
            return json.load(f)

    # Initialize OpenAI client
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )

    if not client.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    classifications = {}

    print(f"Classifying {len(disease_names)} diseases...")

    for i, disease_name in enumerate(disease_names):
        print(f"Processing {i+1}/{len(disease_names)}: {disease_name}")

        result = query_llm_for_histopathology_detectability(disease_name, client)

        if result is not None:
            classifications[disease_name] = result
            print(f"  -> Score: {result}")
        else:
            print(f"  -> Error, skipping")

        # Rate limiting - be respectful to the API
        time.sleep(0.5)

        # Save progress every 10 diseases
        if (i + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(classifications, f, indent=2)
            print(f"Progress saved to {output_path}")

    # Final save
    with open(output_path, "w") as f:
        json.dump(classifications, f, indent=2)

    print(f"Classification complete. Results saved to {output_path}")
    return classifications


def create_histopathology_detectability_plots(
    per_class_data: pd.DataFrame,
    histopathology_classifications: Dict[str, int],
    output_dir: Path,
):
    """
    Create violin plots comparing performance improvements between
    histopathology-detectable and non-detectable diseases
    """
    # Apply matplotlib style
    plt.style.use(snakemake.input.mpl_style)
    # Extract disease names and merge with classifications
    analysis_data = []

    for _, row in per_class_data.iterrows():
        disease_name = extract_disease_name(row["class"])
        if disease_name and disease_name in histopathology_classifications:
            detectability_score = histopathology_classifications[disease_name]

            # Create binary categories for violin plots (scores 5-7 = detectable, 1-4 = not detectable)
            is_detectable = detectability_score >= 5

            # Create more granular categories
            if detectability_score >= 6:
                category = "Highly Detectable (6-7)"
            elif detectability_score >= 5:
                category = "Moderately Detectable (5)"
            elif detectability_score >= 3:
                category = "Uncertain (3-4)"
            else:
                category = "Not Detectable (1-2)"

            analysis_data.append(
                {
                    "class": row["class"],
                    "disease_name": disease_name,
                    "detectability_score": detectability_score,
                    "histopathology_detectable": (
                        "H&E Detectable" if is_detectable else "H&E Non-detectable"
                    ),
                    "detectability_category": category,
                    "f1_improvement": row["f1_improvement"],
                    "rocauc_improvement": row["rocauc_improvement"],
                }
            )

    analysis_df = pd.DataFrame(analysis_data)

    if len(analysis_df) == 0:
        print(
            "No matching diseases found between per-class data and LLM classifications"
        )
        return

    print(f"Created analysis dataset with {len(analysis_df)} diseases")
    print(
        f"H&E Detectable: {(analysis_df['histopathology_detectable'] == 'H&E Detectable').sum()}"
    )
    print(
        f"H&E Non-detectable: {(analysis_df['histopathology_detectable'] == 'H&E Non-detectable').sum()}"
    )

    # Print score distribution
    print(f"\nDetectability score distribution:")
    for score in range(1, 8):
        count = (analysis_df["detectability_score"] == score).sum()
        print(f"  Score {score}: {count} diseases")

    # Create plots with both binary and granular categories
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(
        "Performance Improvements by H&E Histopathology Detectability", fontsize=16
    )

    # Binary F1 improvement violin plot
    sns.violinplot(
        data=analysis_df,
        x="histopathology_detectable",
        y="f1_improvement",
        inner="box",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("F1 Score Improvement (Binary)")
    axes[0, 0].set_xlabel("Disease Category")
    axes[0, 0].set_ylabel("F1 Improvement")
    axes[0, 0].grid(True, alpha=0.3)

    # Add sample sizes
    for i, category in enumerate(["H&E Detectable", "H&E Non-detectable"]):
        count = (analysis_df["histopathology_detectable"] == category).sum()
        y_max = axes[0, 0].get_ylim()[1]
        axes[0, 0].text(
            i,
            y_max * 0.95,
            f"n={count}",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    # Binary ROCAUC improvement violin plot
    sns.violinplot(
        data=analysis_df,
        x="histopathology_detectable",
        y="rocauc_improvement",
        inner="box",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("ROCAUC Improvement (Binary)")
    axes[0, 1].set_xlabel("Disease Category")
    axes[0, 1].set_ylabel("ROCAUC Improvement")
    axes[0, 1].grid(True, alpha=0.3)

    # Add sample sizes
    for i, category in enumerate(["H&E Detectable", "H&E Non-detectable"]):
        count = (analysis_df["histopathology_detectable"] == category).sum()
        y_max = axes[0, 1].get_ylim()[1]
        axes[0, 1].text(
            i,
            y_max * 0.95,
            f"n={count}",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    # Granular F1 improvement violin plot
    sns.violinplot(
        data=analysis_df,
        x="detectability_category",
        y="f1_improvement",
        inner="box",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("F1 Score Improvement (Granular)")
    axes[1, 0].set_xlabel("Detectability Category")
    axes[1, 0].set_ylabel("F1 Improvement")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Granular ROCAUC improvement violin plot
    sns.violinplot(
        data=analysis_df,
        x="detectability_category",
        y="rocauc_improvement",
        inner="box",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("ROCAUC Improvement (Granular)")
    axes[1, 1].set_xlabel("Detectability Category")
    axes[1, 1].set_ylabel("ROCAUC Improvement")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    # Save plots
    plot_path = output_dir / "histopathology_detectability_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plots saved to {plot_path}")

    # Save analysis data
    csv_path = output_dir / "histopathology_detectability_analysis.csv"
    analysis_df.to_csv(csv_path, index=False)
    print(f"Analysis data saved to {csv_path}")

    # Print statistical summary
    print("\n=== STATISTICAL SUMMARY ===")

    # Binary categories
    for category in ["H&E Detectable", "H&E Non-detectable"]:
        category_data = analysis_df[
            analysis_df["histopathology_detectable"] == category
        ]
        print(f"\n{category} (n={len(category_data)}):")

        for metric in ["f1_improvement", "rocauc_improvement"]:
            values = category_data[metric]
            print(f"  {metric}:")
            print(f"    Mean: {values.mean():.4f}")
            print(f"    Median: {values.median():.4f}")
            print(f"    Std: {values.std():.4f}")
            print(f"    Min: {values.min():.4f}")
            print(f"    Max: {values.max():.4f}")

    # Granular categories
    print(f"\n=== GRANULAR CATEGORY ANALYSIS ===")
    for category in analysis_df["detectability_category"].unique():
        category_data = analysis_df[analysis_df["detectability_category"] == category]
        print(f"\n{category} (n={len(category_data)}):")

        for metric in ["f1_improvement", "rocauc_improvement"]:
            values = category_data[metric]
            print(f"  {metric}: mean={values.mean():.4f}, std={values.std():.4f}")

    # Statistical comparison
    detectable_f1 = analysis_df[
        analysis_df["histopathology_detectable"] == "H&E Detectable"
    ]["f1_improvement"]
    non_detectable_f1 = analysis_df[
        analysis_df["histopathology_detectable"] == "H&E Non-detectable"
    ]["f1_improvement"]

    detectable_rocauc = analysis_df[
        analysis_df["histopathology_detectable"] == "H&E Detectable"
    ]["rocauc_improvement"]
    non_detectable_rocauc = analysis_df[
        analysis_df["histopathology_detectable"] == "H&E Non-detectable"
    ]["rocauc_improvement"]

    print(f"\n=== GROUP COMPARISONS ===")
    print(f"F1 Improvement:")
    print(f"  H&E Detectable mean: {detectable_f1.mean():.4f}")
    print(f"  H&E Non-detectable mean: {non_detectable_f1.mean():.4f}")
    print(f"  Difference: {detectable_f1.mean() - non_detectable_f1.mean():.4f}")

    print(f"ROCAUC Improvement:")
    print(f"  H&E Detectable mean: {detectable_rocauc.mean():.4f}")
    print(f"  H&E Non-detectable mean: {non_detectable_rocauc.mean():.4f}")
    print(
        f"  Difference: {detectable_rocauc.mean() - non_detectable_rocauc.mean():.4f}"
    )

    # Correlation analysis
    print(f"\n=== CORRELATION ANALYSIS ===")
    f1_corr = analysis_df["detectability_score"].corr(analysis_df["f1_improvement"])
    rocauc_corr = analysis_df["detectability_score"].corr(
        analysis_df["rocauc_improvement"]
    )
    print(f"Detectability score vs F1 improvement: r={f1_corr:.4f}")
    print(f"Detectability score vs ROCAUC improvement: r={rocauc_corr:.4f}")

    plt.show()

    return analysis_df


def main():
    """Main analysis function"""
    # Load per-class analysis data from snakemake input
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
        rocauc_data[["class", "rocauc_improvement"]],
        f1_data[["class", "f1_improvement"]],
        on="class",
        how="inner",
    )

    print(f"Loaded {len(merged_data)} human disease classes for analysis")

    # Extract unique disease names
    disease_names = []
    for _, row in merged_data.iterrows():
        disease_name = extract_disease_name(row["class"])
        if disease_name:
            disease_names.append(disease_name)

    unique_diseases = list(set(disease_names))
    print(f"Found {len(unique_diseases)} unique diseases")

    # Classify diseases using LLM
    output_dir = Path(snakemake.output.classifications).parent
    classifications_path = Path(snakemake.output.classifications)

    try:
        histopathology_classifications = classify_diseases_by_histopathology(
            unique_diseases, classifications_path
        )

        # Create analysis plots
        create_histopathology_detectability_plots(
            merged_data, histopathology_classifications, output_dir
        )

    except ValueError as e:
        print(f"Error: {e}")
        print("Please set the OPENAI_API_KEY environment variable to run LLM analysis")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return


if __name__ == "__main__":
    main()
