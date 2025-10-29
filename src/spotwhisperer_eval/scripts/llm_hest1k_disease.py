#!/usr/bin/env python3
"""
LLM-based analysis to determine if diseases are likely represented in HEST1K dataset
"""

import pandas as pd
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import openai
import time
from collections import Counter
from huggingface_hub import snapshot_download


def download_hest_metadata():
    """
    Download HEST metadata from HuggingFace repository
    """
    metadata_dir = Path("hest_metadata")

    if metadata_dir.exists() and len(list(metadata_dir.glob("*.json"))) > 0:
        print(f"HEST metadata already exists in {metadata_dir}")
        return metadata_dir

    print("Downloading HEST metadata from HuggingFace...")

    try:
        # Download only the metadata folder from the HEST repository
        snapshot_download(
            repo_id="MahmoodLab/hest",
            repo_type="dataset",
            local_dir=str(metadata_dir.parent),
            allow_patterns="metadata/*.json",
        )

        # Move files from downloaded metadata subfolder to our target directory
        downloaded_metadata_dir = metadata_dir.parent / "metadata"
        if downloaded_metadata_dir.exists():
            metadata_dir.mkdir(exist_ok=True)
            for json_file in downloaded_metadata_dir.glob("*.json"):
                json_file.rename(metadata_dir / json_file.name)
            downloaded_metadata_dir.rmdir()

        print(f"Successfully downloaded HEST metadata to {metadata_dir}")

    except Exception as e:
        print(f"Error downloading HEST metadata: {e}")
        print("You may need to manually download the metadata folder from:")
        print("https://huggingface.co/datasets/MahmoodLab/hest/tree/main/metadata")
        print("and place the JSON files in the 'hest_metadata' directory.")

    return metadata_dir


def load_hest_metadata(metadata_dir: Path) -> List[Dict]:
    """
    Load all JSON metadata files from HEST dataset
    """
    json_files = list(metadata_dir.glob("*.json"))

    if len(json_files) == 0:
        raise FileNotFoundError(
            f"No JSON files found in {metadata_dir}. Please download HEST metadata first."
        )

    print(f"Loading {len(json_files)} metadata files...")

    metadata_list = []
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                metadata = json.load(f)
                metadata_list.append(metadata)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    print(f"Successfully loaded {len(metadata_list)} metadata entries")
    return metadata_list


def aggregate_hest_biological_info(metadata_list: List[Dict], client) -> str:
    """
    Use LLM to aggregate biological/medical information from HEST metadata
    """
    # Extract key biological information
    organs = []
    diseases = []
    disease_states = []
    oncotree_codes = []
    tissues = []
    species = []

    for metadata in metadata_list:
        if "organ" in metadata and metadata["organ"]:
            organs.append(str(metadata["organ"]))
        if "disease_state" in metadata and metadata["disease_state"]:
            disease_states.append(str(metadata["disease_state"]))
        if "oncotree_code" in metadata and metadata["oncotree_code"]:
            oncotree_codes.append(str(metadata["oncotree_code"]))
        if "tissue" in metadata and metadata["tissue"]:
            tissues.append(str(metadata["tissue"]))
        if "species" in metadata and metadata["species"]:
            species.append(str(metadata["species"]))

    # Count occurrences
    organ_counts = Counter(organs)
    disease_state_counts = Counter(disease_states)
    oncotree_counts = Counter(oncotree_codes)
    tissue_counts = Counter(tissues)
    species_counts = Counter(species)

    # Create summary for LLM
    summary_data = {
        "total_samples": len(metadata_list),
        "organs": dict(organ_counts.most_common(20)),
        "disease_states": dict(disease_state_counts.most_common(20)),
        "oncotree_codes": dict(oncotree_counts.most_common(20)),
        "tissues": dict(tissue_counts.most_common(20)),
        "species": dict(species_counts.most_common(10)),
    }

    prompt = f"""
Analyze the following biological and medical information from the HEST1K spatial transcriptomics dataset and provide a comprehensive but concise description of what diseases, tissues, and biological conditions are represented.

Dataset Summary:
- Total samples: {summary_data['total_samples']}
- Organs represented: {summary_data['organs']}
- Disease states: {summary_data['disease_states']}
- OncoTree codes: {summary_data['oncotree_codes']}
- Tissues: {summary_data['tissues']}
- Species: {summary_data['species']}

Please provide a comprehensive description (2-3 paragraphs) that summarizes:
1. The main types of diseases and conditions represented
2. The primary organs and tissues covered
3. The overall scope and focus of the dataset

Focus on the biological and medical content that would be relevant for determining if specific diseases might be represented in this dataset.
"""

    try:
        response = client.chat.completions.create(
            model="claude-4-sonnet",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical expert specializing in spatial transcriptomics and pathology datasets. Provide accurate, comprehensive summaries of biological datasets.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.1,
        )

        description = response.choices[0].message.content.strip()
        return description

    except Exception as e:
        print(f"Error generating HEST description: {e}")
        return None


def query_disease_representation_in_hest(
    disease_name: str, hest_description: str, client
) -> int:
    """
    Query LLM to determine if a disease is likely represented in HEST1K dataset

    Returns integer score from 1-7 (1=very unlikely, 7=very likely)
    """
    prompt = f"""
Based on the following description of the HEST1K spatial transcriptomics dataset, rate how likely the disease "{disease_name}" is to be represented in this dataset.

HEST1K Dataset Description:
{hest_description}

Consider:
- Whether the disease matches the types of conditions represented in the dataset
- Whether the relevant organs/tissues for this disease are included
- Whether the disease would be studied using spatial transcriptomics
- The overall scope and focus of the dataset

Rate on a scale from 1 to 7:
1 = Very unlikely to be represented in HEST1K
2 = Unlikely to be represented in HEST1K
3 = Somewhat unlikely to be represented in HEST1K
4 = Neutral/uncertain
5 = Somewhat likely to be represented in HEST1K
6 = Likely to be represented in HEST1K
7 = Very likely to be represented in HEST1K

Answer with only the number (1-7).
"""

    try:
        response = client.chat.completions.create(
            model="claude-4-sonnet",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical expert specializing in spatial transcriptomics datasets. Provide accurate numerical ratings about disease representation in datasets.",
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


def classify_diseases_by_hest_representation(
    disease_names: List[str], hest_description: str, output_path: Path
) -> Dict[str, int]:
    """
    Classify diseases by their likelihood of representation in HEST1K using LLM

    Returns dictionary mapping disease name to representation score (1-7)
    """
    # Check if results already exist
    if output_path.exists():
        print(f"Loading existing HEST representation classification from {output_path}")
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

    print(f"Classifying {len(disease_names)} diseases for HEST1K representation...")

    for i, disease_name in enumerate(disease_names):
        print(f"Processing {i+1}/{len(disease_names)}: {disease_name}")

        result = query_disease_representation_in_hest(
            disease_name, hest_description, client
        )

        if result is not None:
            classifications[disease_name] = result
            print(f"  -> Score: {result}")
        else:
            print(f"  -> Error, skipping")

        # Rate limiting
        time.sleep(0.5)

        # Save progress every 10 diseases
        if (i + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(classifications, f, indent=2)
            print(f"Progress saved to {output_path}")

    # Final save
    with open(output_path, "w") as f:
        json.dump(classifications, f, indent=2)

    print(
        f"HEST representation classification complete. Results saved to {output_path}"
    )
    return classifications


def create_hest_representation_analysis(
    original_classifications: Dict[str, int],
    hest_classifications: Dict[str, int],
    output_dir: Path,
):
    """
    Create analysis comparing H&E detectability with HEST representation likelihood
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Apply matplotlib style
    plt.style.use(snakemake.input.mpl_style)

    # Merge the two classification datasets
    analysis_data = []

    for disease_name in original_classifications.keys():
        if disease_name in hest_classifications:
            analysis_data.append(
                {
                    "disease_name": disease_name,
                    "he_detectability": original_classifications[disease_name],
                    "hest_representation": hest_classifications[disease_name],
                }
            )

    analysis_df = pd.DataFrame(analysis_data)

    if len(analysis_df) == 0:
        print("No overlapping diseases found between H&E and HEST classifications")
        return

    print(f"Created analysis dataset with {len(analysis_df)} diseases")

    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("H&E Detectability vs HEST1K Representation Analysis", fontsize=16)

    # Scatter plot
    axes[0, 0].scatter(
        analysis_df["he_detectability"], analysis_df["hest_representation"], alpha=0.6
    )
    axes[0, 0].set_xlabel("H&E Detectability Score")
    axes[0, 0].set_ylabel("HEST1K Representation Score")
    axes[0, 0].set_title("H&E Detectability vs HEST Representation")
    axes[0, 0].grid(True, alpha=0.3)

    # Calculate correlation
    correlation = analysis_df["he_detectability"].corr(
        analysis_df["hest_representation"]
    )
    axes[0, 0].text(
        0.05,
        0.95,
        f"r = {correlation:.3f}",
        transform=axes[0, 0].transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Distribution of H&E detectability scores
    axes[0, 1].hist(
        analysis_df["he_detectability"],
        bins=7,
        alpha=0.7,
        color="blue",
        edgecolor="black",
    )
    axes[0, 1].set_xlabel("H&E Detectability Score")
    axes[0, 1].set_ylabel("Number of Diseases")
    axes[0, 1].set_title("Distribution of H&E Detectability Scores")
    axes[0, 1].grid(True, alpha=0.3)

    # Distribution of HEST representation scores
    axes[1, 0].hist(
        analysis_df["hest_representation"],
        bins=7,
        alpha=0.7,
        color="green",
        edgecolor="black",
    )
    axes[1, 0].set_xlabel("HEST1K Representation Score")
    axes[1, 0].set_ylabel("Number of Diseases")
    axes[1, 0].set_title("Distribution of HEST Representation Scores")
    axes[1, 0].grid(True, alpha=0.3)

    # 2D histogram/heatmap
    hist, xedges, yedges = np.histogram2d(
        analysis_df["he_detectability"], analysis_df["hest_representation"], bins=7
    )
    im = axes[1, 1].imshow(
        hist.T, origin="lower", extent=[0.5, 7.5, 0.5, 7.5], cmap="Blues"
    )
    axes[1, 1].set_xlabel("H&E Detectability Score")
    axes[1, 1].set_ylabel("HEST1K Representation Score")
    axes[1, 1].set_title("2D Distribution Heatmap")
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()

    # Save plots
    plot_path = output_dir / "hest_representation_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Analysis plots saved to {plot_path}")

    # Save analysis data
    csv_path = output_dir / "hest_representation_analysis.csv"
    analysis_df.to_csv(csv_path, index=False)
    print(f"Analysis data saved to {csv_path}")

    # Print summary statistics
    print(f"\n=== HEST REPRESENTATION ANALYSIS ===")
    print(f"Total diseases analyzed: {len(analysis_df)}")
    print(
        f"Correlation between H&E detectability and HEST representation: {correlation:.4f}"
    )

    print(f"\nHEST representation score distribution:")
    for score in range(1, 8):
        count = (analysis_df["hest_representation"] == score).sum()
        print(f"  Score {score}: {count} diseases")

    # High HEST representation diseases
    high_hest = analysis_df[analysis_df["hest_representation"] >= 6]
    print(f"\nDiseases with high HEST representation (≥6): {len(high_hest)}")
    for _, row in high_hest.head(10).iterrows():
        print(
            f"  {row['disease_name']}: HEST={row['hest_representation']}, H&E={row['he_detectability']}"
        )

    plt.show()

    return analysis_df


def main():
    """Main analysis function"""
    # Download/load HEST metadata
    metadata_dir = download_hest_metadata()

    try:
        metadata_list = load_hest_metadata(metadata_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please manually download the HEST metadata and try again.")
        return

    # Initialize OpenAI client
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )

    if not client.api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    # Generate HEST dataset description
    hest_description_path = Path(snakemake.output.hest_description)

    if hest_description_path.exists():
        print(f"Loading existing HEST description from {hest_description_path}")
        with open(hest_description_path, "r") as f:
            hest_description = f.read()
    else:
        print("Generating HEST dataset description...")
        hest_description = aggregate_hest_biological_info(metadata_list, client)

        if hest_description:
            with open(hest_description_path, "w") as f:
                f.write(hest_description)
            print(f"HEST description saved to {hest_description_path}")
        else:
            print("Failed to generate HEST description")
            return

    print(f"\nHEST Dataset Description:\n{hest_description}\n")

    # Load disease names from previous analysis
    with open(snakemake.input.histopathology_classifications, "r") as f:
        original_classifications = json.load(f)

    disease_names = list(original_classifications.keys())
    print(f"Found {len(disease_names)} diseases from previous analysis")

    # Classify diseases for HEST representation
    output_dir = Path(snakemake.output.hest_classifications).parent
    hest_classifications_path = Path(snakemake.output.hest_classifications)

    try:
        hest_classifications = classify_diseases_by_hest_representation(
            disease_names, hest_description, hest_classifications_path
        )

        # Create comparative analysis
        create_hest_representation_analysis(
            original_classifications, hest_classifications, output_dir
        )

    except ValueError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return


if __name__ == "__main__":
    main()
