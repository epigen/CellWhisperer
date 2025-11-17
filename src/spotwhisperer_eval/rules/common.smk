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

# Base dataset names
BASE_DATASETS = ["cellxgene_census__archs4_geo", "hest1k", "quilt1m"]

# Generate all dataset combinations (1..n) for wildcards
DATASET_COMBOS = []
for r in range(1, len(BASE_DATASETS) + 1):
    for combo in itertools.combinations(BASE_DATASETS, r):
        combo_name = "__".join(sorted(combo))
        DATASET_COMBOS.append(combo_name)

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
