#!/usr/bin/env python
"""
Aggregate retrieval analysis results from multiple test datasets
"""
import pandas as pd


def rename_fn(col):
    return (
        col.replace("transcriptome_text", "left_right")
        .replace("text_transcriptome", "right_left")
        .replace("image_text,", "left_right")
        .replace("text_image", "right_left")
        .replace("image_tex", "left_right")
        .replace("text_image", "right_left")
        .replace("transcriptome_image", "left_right")
        .replace("image_transcriptome", "right_left")
    )


dfs = []
for dataset, fn in zip(snakemake.params.test_datasets, snakemake.input):
    df = pd.read_csv(fn)
    df.columns = [rename_fn(col) for col in df.columns]
    df["test_dataset"] = dataset
    dfs.append(
        df.iloc[-1]
    )  # the last row represents the full result, the others are batch-level scores

df = pd.DataFrame(dfs).set_index("test_dataset")

retrieval_cols = [col for col in df.columns if col.startswith("test")]
df[retrieval_cols].to_csv(snakemake.output.aggregated_retrieval)

cweval_cols = [col for col in df.columns if not col.startswith("test")]

print(
    (df[cweval_cols] == df[cweval_cols].iloc[0]).all()
)  # check that most (except loss etc., because of the shuffling) are the same across the different runs.

df[cweval_cols].iloc[0].to_csv(snakemake.output.aggregated_cwevals)
