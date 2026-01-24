#!/usr/bin/env python3
"""
Split baseline logits CSV (containing all datasets) into per-dataset files
that match the expected "scores" schema for downstream metrics.

This rule only performs splitting; it does not run any model scoring.

Inputs (Snakemake):
- snakemake.input.baseline_csv: path to baseline logits CSV (terms1)

Outputs:
- snakemake.output.score: single per-dataset CSV under
  results/pathocell_evaluation/{baseline_model}/{dataset}_{prediction_level}_scores_seed{seed}.csv

Notes:
- We write only the class score columns (exclude identifier columns).
- Column names may be numeric (0..N-1); downstream metrics align them to class names.
"""

from pathlib import Path
import re
import pandas as pd

baseline_fp = Path(snakemake.input.baseline_csv)
out_fp = Path(snakemake.output.score)
prediction_level = snakemake.wildcards.prediction_level

# Read baseline logits
df = pd.read_csv(baseline_fp)
# Keep only patch rows
if "source_image" in df.columns:
    df = df[df["source_image"].str.contains("_patch.tiff")].copy()

def parse_dataset_id(x: str) -> str:
    m = re.search(r"(reg\d+_[AB])", x)
    return m.group(1) if m else x

if "dataset_id" not in df.columns:
    # derive from source_image if present
    if "source_image" in df.columns:
        df["dataset_id"] = df["source_image"].apply(parse_dataset_id)
    else:
        raise KeyError("Cannot determine dataset_id from baseline CSV")

# Identify class columns: exclude id columns
id_cols = {"source_image", "spot_id", "dataset_id"}
class_cols = [c for c in df.columns if c not in id_cols]

# Write one file per dataset x seed according to outputs list
# Extract dataset from wildcard
dataset = snakemake.wildcards.dataset

sub = df[df["dataset_id"] == dataset]
sub[class_cols].to_csv(out_fp, index=False)
