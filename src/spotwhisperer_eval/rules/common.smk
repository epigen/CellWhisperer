# Common definitions and shared constants for SpotWhisperer eval

from pathlib import Path
import itertools
import glob

# Define modality colors
MODALITY_COLORS = {
    "text": "#33aac3",
    "image": "#8c1515",
    "transcriptome": "#9c7cb8",
    "text-image": "#e74c3c",
    "image-transcriptome": "#3498db",
    "text-transcriptome": "#2ecc71",
    "trimodal": "#f39c12",
}

# Seeds for reproducibility
SEEDS = [0]

# Individual datasets that actually exist (or will exist)
INDIVIDUAL_DATASETS = ["cellxgene_census", "archs4_geo", "hest1k", "quilt1m"]

# Classify datasets by their file structure
# Multi-file datasets use the h5ads/ directory structure with multiple files
# Single-file datasets use a single full_data.h5ad file
MULTI_FILE_DATASETS = ["hest1k", "quilt1m"]  # datasets that have h5ads/ directory structure
SINGLE_FILE_DATASETS = [d for d in INDIVIDUAL_DATASETS if d not in MULTI_FILE_DATASETS]

# Define modality pairs for logical groupings
MODALITY_PAIRS = {
    "transcriptome_text": ["cellxgene_census", "archs4_geo"], 
    "transcriptome_image": ["hest1k"],
    "image_text": ["quilt1m"]
}

# Generate original dataset combinations using existing logic  
ORIGINAL_BASE_DATASETS = ["cellxgene_census__archs4_geo", "hest1k", "quilt1m"]
# Keep BASE_DATASETS for backward compatibility with existing rules
BASE_DATASETS = ORIGINAL_BASE_DATASETS
DATASET_COMBOS = []
for r in range(1, len(ORIGINAL_BASE_DATASETS) + 1):
    for combo in itertools.combinations(ORIGINAL_BASE_DATASETS, r):
        combo_name = "__".join(sorted(combo))
        DATASET_COMBOS.append(combo_name)

# Define sampling scenarios with individual dataset subsampling
SAMPLING_SCENARIOS = {
    # Bimodal with 1/8th data for each modality pair
    "bimodal_eighth": [
        # 1/8th transcriptome-text: subsample both cellxgene_census and archs4_geo
        "cellxgene_census_8thsub__archs4_geo_8thsub",
        # 1/8th transcriptome-image
        "hest1k_8thsub", 
        # 1/8th image-text
        "quilt1m_8thsub"
    ],
    
    # Trimodal with 1/8th of one modality pair, full for others
    "trimodal_partial": [
        # 1/8th transcriptome-text + full others
        "cellxgene_census_8thsub__archs4_geo_8thsub__hest1k__quilt1m",
        # 1/8th transcriptome-image + full others
        "cellxgene_census__archs4_geo__hest1k_8thsub__quilt1m",
        # 1/8th image-text + full others
        "cellxgene_census__archs4_geo__hest1k__quilt1m_8thsub"
    ],
    
    # Poor pairs: 1/8th of 2 modality pairs, full for 1
    "poor_pairs": [
        # 1/8th transcriptome-text + 1/8th transcriptome-image + full image-text
        "cellxgene_census_8thsub__archs4_geo_8thsub__hest1k_8thsub__quilt1m",
        # 1/8th transcriptome-text + 1/8th image-text + full transcriptome-image
        "cellxgene_census_8thsub__archs4_geo_8thsub__hest1k__quilt1m_8thsub", 
        # 1/8th transcriptome-image + 1/8th image-text + full transcriptome-text
        "cellxgene_census__archs4_geo__hest1k_8thsub__quilt1m_8thsub"
    ],
    
    # Poor trimodal: 1/8th of all modality pairs
    "poor_trimodal": [
        "cellxgene_census_8thsub__archs4_geo_8thsub__hest1k_8thsub__quilt1m_8thsub"
    ]
}

# Extend DATASET_COMBOS to include all sampling scenarios
for scenario_combos in SAMPLING_SCENARIOS.values():
    DATASET_COMBOS.extend(scenario_combos)

# Model mapping for each test dataset
MODEL_MAPPINGS = {
    "immgen": {
        "naive_baseline": "hest1k",
        "bimodal_matching": "cellxgene_census__archs4_geo",
        "bimodal_bridge": "hest1k__quilt1m",
        "trimodal": "cellxgene_census__archs4_geo__hest1k__quilt1m",
    },
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

# Dataset pair to modality mapping
DATASET_PAIR_MAPPING = {
    "cellxgene_census": "transcriptome-text",
    "archs4_geo": "transcriptome-text",
    "cellxgene_census__archs4_geo": "transcriptome-text",
    "hest1k": "transcriptome-image",
    "quilt1m": "image-text",
}

# Results directories
SPOTWHISPERER_EVAL_RESULTS = PROJECT_DIR / "results/spotwhisperer_eval"
MODELS_DIR = SPOTWHISPERER_EVAL_RESULTS / "models"
BENCHMARKS_DIR = SPOTWHISPERER_EVAL_RESULTS / "benchmarks"
