#!/usr/bin/env python
"""
Write a combined manifest after ensuring the three per-pair manifests exist.
Inputs:
- snakemake.input: text, image_transcriptome, image_text manifest paths
- snakemake.output.combined_manifest: target path
"""
from pathlib import Path

from pathlib import Path

Path(snakemake.output.combined_manifest).parent.mkdir(parents=True, exist_ok=True)
ratio = snakemake.wildcards.subratio
with open(snakemake.output.combined_manifest, "w") as fh:
    fh.write("\n".join([
        f"subset_performance/transcriptome-text/{ratio}/comparison.png",
        f"subset_performance/transcriptome-image/{ratio}/comparison.png",
        f"subset_performance/image-text/{ratio}/comparison.png",
    ]))
