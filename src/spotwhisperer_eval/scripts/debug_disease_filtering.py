#!/usr/bin/env python3
"""
Debug script to identify where diseases are being filtered out in the analysis pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import Counter


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
        print(f"WARNING: Trimodal results not found at {trimodal_path}")
        return None, None

    trimodal_df = pd.read_csv(trimodal_path)
    print(f"Loaded trimodal results: {len(trimodal_df)} classes")

    # Load bimodal matching results
    bimodal_path = Path(
        "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/spotwhisperer_eval/benchmarks/cellwhisperer/spotwhisperer_cellxgene_census__archs4_geo/datasets/human_disease/celltype/performance_metrics_permetadataraw.csv"
    )

    if not bimodal_path.exists():
        print(f"WARNING: Bimodal results not found at {bimodal_path}")
        return trimodal_df, None

    bimodal_df = pd.read_csv(bimodal_path)
    print(f"Loaded bimodal results: {len(bimodal_df)} classes")

    return trimodal_df, bimodal_df


def load_detectability_scores():
    """Load H&E detectability scores from previous analysis"""

    detectability_path = Path("hest_representation_analysis.csv")
    if not detectability_path.exists():
        print(f"WARNING: Detectability scores not found at {detectability_path}")
        return None

    detectability_df = pd.read_csv(detectability_path)
    print(f"Loaded detectability scores for {len(detectability_df)} diseases")

    return detectability_df


def load_per_class_analysis():
    """Load per-class analysis data"""

    per_class_path = Path("per_class_analysis.csv")
    if not per_class_path.exists():
        print(f"WARNING: Per-class analysis not found at {per_class_path}")
        return None

    per_class_df = pd.read_csv(per_class_path)
    print(f"Loaded per-class analysis: {len(per_class_df)} entries")

    return per_class_df


def debug_filtering_steps():
    """Debug each step of the filtering process"""

    print("=" * 60)
    print("DEBUGGING DISEASE FILTERING PIPELINE")
    print("=" * 60)

    # Step 1: Load all datasets
    print("\n1. LOADING ALL DATASETS")
    print("-" * 30)

    trimodal_df, bimodal_df = load_original_evaluation_results()
    detectability_df = load_detectability_scores()
    per_class_df = load_per_class_analysis()

    if detectability_df is None:
        print("ERROR: Cannot proceed without detectability scores")
        return

    # Step 2: Analyze detectability dataset
    print("\n2. ANALYZING DETECTABILITY DATASET")
    print("-" * 30)
    print(f"Total diseases in detectability analysis: {len(detectability_df)}")
    print(f"Sample diseases: {list(detectability_df['disease_name'].head(10))}")

    # Step 3: Analyze original evaluation results
    print("\n3. ANALYZING ORIGINAL EVALUATION RESULTS")
    print("-" * 30)

    if trimodal_df is not None:
        print(f"Trimodal classes: {len(trimodal_df)}")
        print(f"Sample trimodal classes: {list(trimodal_df['class'].head(5))}")

        # Extract disease names from trimodal
        trimodal_diseases = []
        trimodal_failed_extractions = []

        for class_name in trimodal_df["class"]:
            disease_name = extract_disease_name(class_name)
            if disease_name:
                trimodal_diseases.append(disease_name)
            else:
                trimodal_failed_extractions.append(class_name)

        print(
            f"Successfully extracted {len(trimodal_diseases)} disease names from trimodal"
        )
        print(
            f"Failed to extract {len(trimodal_failed_extractions)} disease names from trimodal"
        )

        if trimodal_failed_extractions:
            print(f"Sample failed extractions: {trimodal_failed_extractions[:5]}")

        trimodal_unique_diseases = list(set(trimodal_diseases))
        print(f"Unique diseases in trimodal: {len(trimodal_unique_diseases)}")

    if bimodal_df is not None:
        print(f"Bimodal classes: {len(bimodal_df)}")
        print(f"Sample bimodal classes: {list(bimodal_df['class'].head(5))}")

        # Extract disease names from bimodal
        bimodal_diseases = []
        bimodal_failed_extractions = []

        for class_name in bimodal_df["class"]:
            disease_name = extract_disease_name(class_name)
            if disease_name:
                bimodal_diseases.append(disease_name)
            else:
                bimodal_failed_extractions.append(class_name)

        print(
            f"Successfully extracted {len(bimodal_diseases)} disease names from bimodal"
        )
        print(
            f"Failed to extract {len(bimodal_failed_extractions)} disease names from bimodal"
        )

        bimodal_unique_diseases = list(set(bimodal_diseases))
        print(f"Unique diseases in bimodal: {len(bimodal_unique_diseases)}")

    # Step 4: Check overlap between trimodal and bimodal
    if trimodal_df is not None and bimodal_df is not None:
        print("\n4. CHECKING TRIMODAL-BIMODAL OVERLAP")
        print("-" * 30)

        # Check class overlap
        trimodal_classes = set(trimodal_df["class"])
        bimodal_classes = set(bimodal_df["class"])

        common_classes = trimodal_classes.intersection(bimodal_classes)
        trimodal_only = trimodal_classes - bimodal_classes
        bimodal_only = bimodal_classes - trimodal_classes

        print(f"Classes in both trimodal and bimodal: {len(common_classes)}")
        print(f"Classes only in trimodal: {len(trimodal_only)}")
        print(f"Classes only in bimodal: {len(bimodal_only)}")

        if trimodal_only:
            print(f"Sample trimodal-only classes: {list(trimodal_only)[:5]}")
        if bimodal_only:
            print(f"Sample bimodal-only classes: {list(bimodal_only)[:5]}")

        # Check disease overlap
        trimodal_diseases_set = set(trimodal_unique_diseases)
        bimodal_diseases_set = set(bimodal_unique_diseases)

        common_diseases = trimodal_diseases_set.intersection(bimodal_diseases_set)
        print(f"Diseases in both trimodal and bimodal: {len(common_diseases)}")

    # Step 5: Check overlap with detectability scores
    print("\n5. CHECKING OVERLAP WITH DETECTABILITY SCORES")
    print("-" * 30)

    detectability_diseases = set(detectability_df["disease_name"])

    if trimodal_df is not None:
        trimodal_detectability_overlap = trimodal_diseases_set.intersection(
            detectability_diseases
        )
        trimodal_not_in_detectability = trimodal_diseases_set - detectability_diseases
        detectability_not_in_trimodal = detectability_diseases - trimodal_diseases_set

        print(
            f"Trimodal diseases also in detectability: {len(trimodal_detectability_overlap)}"
        )
        print(
            f"Trimodal diseases NOT in detectability: {len(trimodal_not_in_detectability)}"
        )
        print(
            f"Detectability diseases NOT in trimodal: {len(detectability_not_in_trimodal)}"
        )

        if trimodal_not_in_detectability:
            print(
                f"Sample trimodal diseases not in detectability: {list(trimodal_not_in_detectability)[:10]}"
            )

        if detectability_not_in_trimodal:
            print(
                f"Sample detectability diseases not in trimodal: {list(detectability_not_in_trimodal)[:10]}"
            )

    # Step 6: Analyze per-class analysis if available
    if per_class_df is not None:
        print("\n6. ANALYZING PER-CLASS ANALYSIS")
        print("-" * 30)

        # Filter for human_disease dataset
        human_disease_df = per_class_df[per_class_df["dataset"] == "human_disease"]
        print(f"Human disease entries in per-class analysis: {len(human_disease_df)}")

        # Get unique classes
        per_class_classes = set(human_disease_df["class"])
        print(f"Unique classes in per-class human_disease: {len(per_class_classes)}")

        # Extract disease names
        per_class_diseases = []
        per_class_failed_extractions = []

        for class_name in per_class_classes:
            disease_name = extract_disease_name(class_name)
            if disease_name:
                per_class_diseases.append(disease_name)
            else:
                per_class_failed_extractions.append(class_name)

        per_class_unique_diseases = list(set(per_class_diseases))
        print(
            f"Unique diseases in per-class analysis: {len(per_class_unique_diseases)}"
        )
        print(f"Failed extractions in per-class: {len(per_class_failed_extractions)}")

        # Check overlap with detectability
        per_class_diseases_set = set(per_class_unique_diseases)
        per_class_detectability_overlap = per_class_diseases_set.intersection(
            detectability_diseases
        )

        print(
            f"Per-class diseases also in detectability: {len(per_class_detectability_overlap)}"
        )

        # This should be our final number
        print(
            f"FINAL OVERLAP (per-class ∩ detectability): {len(per_class_detectability_overlap)}"
        )

    # Step 7: Simulate the high_detectability_disease_analysis.py merge
    if trimodal_df is not None and bimodal_df is not None:
        print("\n7. SIMULATING HIGH_DETECTABILITY_DISEASE_ANALYSIS MERGE")
        print("-" * 30)

        # Merge trimodal and bimodal results on class
        merged_df = pd.merge(
            trimodal_df[["class", "f1", "n_samples_in_class"]],
            bimodal_df[["class", "f1"]],
            on="class",
            suffixes=("_trimodal", "_bimodal"),
        )

        print(f"After trimodal-bimodal merge: {len(merged_df)} classes")

        # Extract disease names and merge with detectability scores
        merged_df["disease_name"] = merged_df["class"].apply(extract_disease_name)

        # Remove rows where disease name extraction failed
        merged_df = merged_df.dropna(subset=["disease_name"])
        print(f"After disease name extraction: {len(merged_df)} classes")

        # Merge with detectability scores
        final_df = pd.merge(
            merged_df,
            detectability_df[["disease_name", "he_detectability"]],
            on="disease_name",
            how="inner",
        )

        print(f"After detectability merge: {len(final_df)} diseases")
        print(f"This should match the ~70 number you're seeing!")

        # Show some examples of what made it through
        print(f"\nSample diseases that made it through the full pipeline:")
        for disease in final_df["disease_name"].head(10):
            print(f"  - {disease}")

    # Step 8: Identify the main bottleneck
    print("\n8. IDENTIFYING THE MAIN BOTTLENECK")
    print("-" * 30)

    if trimodal_df is not None and bimodal_df is not None:
        print("The main filtering steps are:")
        print(f"1. Original detectability diseases: {len(detectability_df)}")
        print(f"2. Trimodal classes: {len(trimodal_df)}")
        print(f"3. Bimodal classes: {len(bimodal_df)}")
        print(f"4. Classes in both trimodal AND bimodal: {len(common_classes)}")
        print(f"5. Diseases extractable from common classes: {len(final_df)}")

        # Calculate loss at each step
        loss_at_merge = len(detectability_df) - len(common_classes)
        loss_at_extraction = len(common_classes) - len(final_df)

        print(f"\nLoss analysis:")
        print(f"- Lost at trimodal-bimodal merge: {loss_at_merge}")
        print(
            f"- Lost at disease name extraction/detectability merge: {loss_at_extraction}"
        )

        if loss_at_merge > loss_at_extraction:
            print(
                "\n*** MAIN ISSUE: Many diseases in detectability analysis are not present in BOTH trimodal and bimodal evaluation results ***"
            )
        else:
            print(
                "\n*** MAIN ISSUE: Disease name extraction or detectability matching is failing ***"
            )


def check_disease_name_patterns():
    """Check what patterns exist in the class names"""

    print("\n" + "=" * 60)
    print("ANALYZING CLASS NAME PATTERNS")
    print("=" * 60)

    trimodal_df, bimodal_df = load_original_evaluation_results()

    if trimodal_df is not None:
        print(f"\nAll trimodal class names ({len(trimodal_df)}):")
        for i, class_name in enumerate(trimodal_df["class"]):
            print(f"  {i+1:3d}. {class_name}")
            if i >= 20:  # Show first 20
                print(f"  ... and {len(trimodal_df) - 21} more")
                break

        # Check for different patterns
        patterns = {"healthy_individual": 0, "other_patterns": []}

        for class_name in trimodal_df["class"]:
            if "from a healthy individual" in class_name:
                patterns["healthy_individual"] += 1
            else:
                patterns["other_patterns"].append(class_name)

        print(f"\nPattern analysis:")
        print(
            f"- Classes with 'from a healthy individual': {patterns['healthy_individual']}"
        )
        print(f"- Classes with other patterns: {len(patterns['other_patterns'])}")

        if patterns["other_patterns"]:
            print(f"Sample other patterns:")
            for pattern in patterns["other_patterns"][:10]:
                print(f"  - {pattern}")


def main():
    """Main debugging function"""

    debug_filtering_steps()
    check_disease_name_patterns()

    print("\n" + "=" * 60)
    print("DEBUGGING COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check if the evaluation was actually run on all 230 diseases")
    print("2. Verify that the class naming pattern is consistent")
    print("3. Check if there are missing evaluation result files")
    print(
        "4. Consider if the detectability analysis included diseases not in the evaluation"
    )


if __name__ == "__main__":
    main()
