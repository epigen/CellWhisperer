#!/usr/bin/env python3
"""
Split lung baseline logits CSV (all samples combined) into per-sample files.

The baseline CSV uses source_image values like 'lc_1', 'lc_2', etc.
The sample wildcard matches these directly.
Only class score columns are written (source_image and spot_id are excluded).
"""

from pathlib import Path
import pandas as pd

baseline_fp = Path(snakemake.input.baseline_csv)
out_fp = Path(snakemake.output.score)
sample = snakemake.wildcards.sample  # e.g. 'lc_1'

df = pd.read_csv(baseline_fp)

id_cols = {"source_image", "spot_id"}
class_cols = [c for c in df.columns if c not in id_cols]

sub = df[df["source_image"] == sample]
sub[class_cols].to_csv(out_fp, index=False)
