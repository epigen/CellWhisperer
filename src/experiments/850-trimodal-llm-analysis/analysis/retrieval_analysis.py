#!/usr/bin/env python
"""Aggregate retrieval-level CLIP scores with sample metadata.

This script aligns the retrieval evaluation outputs with the provided sample
metadata for each modality (image-text, transcriptome-image, and
transcriptome-text). It annotates high-CLIP pairs, aggregates summary
statistics, and exports merged tables for downstream biological review.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_ROOT = PROJECT_ROOT / "docs" / "per_class_analysis"
OUTPUT_ROOT = BASE_DIR / "outputs" / "retrieval"


@dataclass
class RetrievalResult:
    full_table: pd.DataFrame
    high_threshold: float
    summary_tables: Dict[str, pd.DataFrame]


def compute_threshold(series: pd.Series, quantile: float) -> float:
    """Return the quantile-based threshold for CLIP scores."""

    if series.empty:
        raise ValueError("Cannot compute threshold on empty series.")
    return float(series.quantile(quantile))


def _flag_high_scores(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["is_high_clip"] = df["avg_clip_score"] >= threshold
    return df


def _export_tables(tables: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def process_image_text(data_root: Path, quantile: float) -> RetrievalResult:
    """Join Quilt-1M retrieval scores with patch metadata."""

    retrieval_path = data_root / "image-text" / "quilt1m_per_class_analysis_retrieval.csv"
    metadata_path = data_root / "image-text" / "quilt1m_sample_metadata.csv"

    retrieval = pd.read_csv(retrieval_path)
    metadata = pd.read_csv(
        metadata_path,
        usecols=["patch_id", "natural_language_annotation", "sample_id", "dataset"],
    )

    merged = retrieval.merge(
        metadata,
        left_on="orig_ids",
        right_on="patch_id",
        how="left",
    )

    threshold = compute_threshold(merged["avg_clip_score"], quantile)
    flagged = _flag_high_scores(merged, threshold)

    sample_summary = (
        flagged.groupby("sample_id")
        .agg(
            high_clip_fraction=("is_high_clip", "mean"),
            high_clip_count=("is_high_clip", "sum"),
            total_patches=("is_high_clip", "size"),
            mean_clip_score=("avg_clip_score", "mean"),
        )
        .reset_index()
        .sort_values("high_clip_fraction", ascending=False)
    )

    def _concat_examples(series: pd.Series, max_items: int = 3) -> str:
        uniques = [s for s in pd.Series(series).dropna().unique() if isinstance(s, str)]
        return " | ".join(uniques[:max_items]) if uniques else ""

    sample_metadata = (
        metadata.groupby("sample_id")
        .agg(
            dataset=("dataset", lambda s: s.dropna().iloc[0] if not s.dropna().empty else None),
            annotation_examples=("natural_language_annotation", _concat_examples),
            num_unique_annotations=("natural_language_annotation", "nunique"),
            num_patches=("patch_id", "count"),
        )
        .reset_index()
    )

    sample_share = sample_summary.merge(sample_metadata, on="sample_id", how="left")

    dataset_summary = (
        flagged.groupby("dataset")
        .agg(
            high_clip_fraction=("is_high_clip", "mean"),
            high_clip_count=("is_high_clip", "sum"),
            total_patches=("is_high_clip", "size"),
            mean_clip_score=("avg_clip_score", "mean"),
        )
        .reset_index()
        .sort_values("high_clip_fraction", ascending=False)
    )

    summary_tables = {
        "quilt1m_full": flagged,
        "quilt1m_sample_summary": sample_summary,
        "quilt1m_sample_summary_share": sample_share,
        "quilt1m_dataset_summary": dataset_summary,
    }

    return RetrievalResult(flagged, threshold, summary_tables)


def process_transcriptome_image(data_root: Path, quantile: float) -> RetrievalResult:
    """Aggregate HEST1K retrieval scores by sample metadata."""

    retrieval_path = data_root / "transcriptome-image" / "hest1k_per_class_analysis_retrieval.csv"
    metadata_path = data_root / "transcriptome-image" / "hest1k_sample_metadata.csv"

    retrieval = pd.read_csv(retrieval_path)
    metadata = pd.read_csv(
        metadata_path,
        usecols=[
            "Unnamed: 0",
            "sample_id",
            "dataset",
            "in_tissue",
            "array_row",
            "array_col",
            "n_genes_by_counts",
            "total_counts",
        ],
    ).rename(columns={"Unnamed: 0": "orig_ids"})

    merged = retrieval.merge(metadata, on="orig_ids", how="left")

    threshold = compute_threshold(merged["avg_clip_score"], quantile)
    flagged = _flag_high_scores(merged, threshold)

    sample_summary = (
        flagged.groupby("sample_id")
        .agg(
            high_clip_fraction=("is_high_clip", "mean"),
            high_clip_count=("is_high_clip", "sum"),
            total_cells=("is_high_clip", "size"),
            mean_clip_score=("avg_clip_score", "mean"),
            mean_gene_counts=("n_genes_by_counts", "mean"),
        )
        .reset_index()
        .sort_values("high_clip_fraction", ascending=False)
    )

    sample_metadata = (
        metadata.groupby("sample_id")
        .agg(
            dataset=("dataset", lambda s: s.dropna().iloc[0] if not s.dropna().empty else None),
            mean_in_tissue=("in_tissue", "mean"),
            median_array_row=("array_row", "median"),
            median_array_col=("array_col", "median"),
            total_cells_raw=("orig_ids", "count"),
        )
        .reset_index()
    )

    sample_share = sample_summary.merge(sample_metadata, on="sample_id", how="left")

    dataset_summary = (
        flagged.groupby("dataset")
        .agg(
            high_clip_fraction=("is_high_clip", "mean"),
            high_clip_count=("is_high_clip", "sum"),
            total_cells=("is_high_clip", "size"),
            mean_clip_score=("avg_clip_score", "mean"),
        )
        .reset_index()
        .sort_values("high_clip_fraction", ascending=False)
    )

    summary_tables = {
        "hest1k_full": flagged,
        "hest1k_sample_summary": sample_summary,
        "hest1k_sample_summary_share": sample_share,
        "hest1k_dataset_summary": dataset_summary,
    }

    return RetrievalResult(flagged, threshold, summary_tables)


def _split_archs4_and_cellxgene(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split retrieval entries into ARCHS4/Geo vs CellxGene cohorts."""

    is_cellxgene = df["orig_ids"].str.startswith("census_")
    return df.loc[~is_cellxgene].copy(), df.loc[is_cellxgene].copy()


def process_transcriptome_text(data_root: Path, quantile: float) -> RetrievalResult:
    """Join transcriptome-text retrieval scores with ARCHS4 and CellxGene metadata."""

    retrieval_path = (
        data_root
        / "transcriptome-text"
        / "cellxgene_census_archs4_geo_per_class_analysis_retrieval.csv"
    )

    retrieval = pd.read_csv(retrieval_path)
    archs4_df, cellx_df = _split_archs4_and_cellxgene(retrieval)

    archs4_metadata = pd.read_csv(
        data_root / "transcriptome-text" / "archs4_geo_sample_metadata.csv",
        usecols=[
            "experiment",
            "sample",
            "biosample_title",
            "natural_language_annotation",
            "mapped_ontology_terms",
            "dataset",
        ],
    ).rename(columns={"experiment": "orig_ids"})

    cellx_metadata = pd.read_csv(
        data_root / "transcriptome-text" / "cellxgene_census_sample_metadata.csv",
        usecols=[
            "cell_id",
            "cell_type",
            "tissue",
            "organ",
            "dataset",
            "natural_language_annotation",
        ],
    ).rename(columns={"cell_id": "orig_ids"})

    archs4_merged = archs4_df.merge(archs4_metadata, on="orig_ids", how="left")
    cellx_merged = cellx_df.merge(cellx_metadata, on="orig_ids", how="left")

    combined = pd.concat([archs4_merged, cellx_merged], ignore_index=True)

    threshold = compute_threshold(combined["avg_clip_score"], quantile)
    flagged = _flag_high_scores(combined, threshold)

    dataset_summary = (
        flagged.groupby("dataset")
        .agg(
            high_clip_fraction=("is_high_clip", "mean"),
            high_clip_count=("is_high_clip", "sum"),
            total_samples=("is_high_clip", "size"),
            mean_clip_score=("avg_clip_score", "mean"),
        )
        .reset_index()
        .sort_values("high_clip_fraction", ascending=False)
    )

    cell_type_summary = (
        flagged.groupby("cell_type")
        .agg(
            high_clip_fraction=("is_high_clip", "mean"),
            high_clip_count=("is_high_clip", "sum"),
            total_samples=("is_high_clip", "size"),
            mean_clip_score=("avg_clip_score", "mean"),
        )
        .reset_index()
        .sort_values("high_clip_fraction", ascending=False)
    )

    def _metadata_examples(df: pd.DataFrame, key: str) -> pd.Series:
        def _concat(values: Iterable[str], max_items: int = 3) -> str:
            uniques = [v for v in pd.Series(values).dropna().unique() if isinstance(v, str)]
            return " | ".join(uniques[:max_items]) if uniques else ""

        return (
            df.groupby(key)["natural_language_annotation"].apply(lambda s: _concat(s, 3)).rename("annotation_examples")
        )

    cell_type_examples = _metadata_examples(flagged, "cell_type") if "cell_type" in flagged.columns else pd.Series(dtype=str)
    dataset_examples = _metadata_examples(flagged, "dataset") if "dataset" in flagged.columns else pd.Series(dtype=str)

    cell_type_share = cell_type_summary.merge(cell_type_examples, left_on="cell_type", right_index=True, how="left")
    dataset_share = dataset_summary.merge(dataset_examples, on="dataset", how="left")

    summary_tables = {
        "transcriptome_text_full": flagged,
        "transcriptome_text_dataset_summary": dataset_summary,
        "transcriptome_text_cell_type_summary": cell_type_summary,
        "transcriptome_text_dataset_summary_share": dataset_share,
        "transcriptome_text_cell_type_summary_share": cell_type_share,
    }

    return RetrievalResult(flagged, threshold, summary_tables)


def write_share_payload(
    output_root: Path,
    *,
    top_n: int = 30,
) -> Path:
    """Create JSON payload with top-N share summaries and modality-specific prompts."""

    payload: Dict[str, Dict[str, object]] = {}

    prompts: Dict[str, str] = {
        "image_text": (
            "You are analyzing high-CLIP histopathology patch ↔ text pairs. For each entry, "
            "use the textual annotation, sample_id, and high_clip_fraction to infer shared "
            "biological or technical patterns (stain type, organ/site, annotation style). "
            "Cluster related items and summarize why these patches align strongly with text."
        ),
        "transcriptome_image": (
            "Review high-CLIP single-cell ↔ histology matches. Using sample_id, gene counts, "
            "and high_clip_fraction, deduce which tissue sections or technical qualities drive "
            "strong alignment. Highlight spatial or biological themes to investigate."
        ),
        "transcriptome_text_cell_types": (
            "Inspect cell types with strong CLIP similarity between RNA profiles and textual "
            "descriptions. Group entries by organ/system or differentiation state. Flag notable "
            "biological themes and suggest follow-up metadata to validate."
        ),
        "transcriptome_text_datasets": (
            "Compare datasets with high CLIP alignment between transcriptomic samples and text. "
            "Identify shared technical or biological factors and propose validation steps."
        ),
    }

    share_configs = {
        "image_text": output_root / "image-text" / "quilt1m_sample_summary_share.csv",
        "transcriptome_image": output_root / "transcriptome-image" / "hest1k_sample_summary_share.csv",
        "transcriptome_text_cell_types": output_root
        / "transcriptome-text"
        / "transcriptome_text_cell_type_summary_share.csv",
        "transcriptome_text_datasets": output_root
        / "transcriptome-text"
        / "transcriptome_text_dataset_summary_share.csv",
    }

    for key, csv_path in share_configs.items():
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        threshold_path = csv_path.parent / "threshold.txt"
        threshold = None
        if threshold_path.exists():
            try:
                threshold = float(threshold_path.read_text().strip())
            except ValueError:
                threshold = None

        payload[key] = {
            "threshold": threshold,
            "columns": df.columns.tolist(),
            "top_entries": df.head(top_n).to_dict(orient="records"),
            "prompt": prompts.get(key, ""),
        }

    output_path = output_root / "share_payload.json"
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def run_all(data_root: Path, output_root: Path, quantile: float) -> Dict[str, RetrievalResult]:
    """Execute retrieval aggregation for all modalities."""

    output_root.mkdir(parents=True, exist_ok=True)

    processors = {
        "image-text": process_image_text,
        "transcriptome-image": process_transcriptome_image,
        "transcriptome-text": process_transcriptome_text,
    }

    results: Dict[str, RetrievalResult] = {}

    for modality, processor in processors.items():
        result = processor(data_root, quantile)
        modality_dir = output_root / modality.replace("/", "_")
        _export_tables(result.summary_tables, modality_dir)
        results[modality] = result

        (modality_dir / "threshold.txt").write_text(f"{result.high_threshold:.6f}\n")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze retrieval CLIP scores with metadata.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Base directory containing retrieval CSVs and metadata.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory to store merged tables and summaries.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Quantile used to flag high CLIP scores (default: 0.95).",
    )
    parser.add_argument(
        "--no-share-payload",
        action="store_true",
        help="Skip generating share_payload.json",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Top N rows to include in the share payload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_all(args.data_root, args.output_root, args.quantile)

    if not args.no_share_payload:
        payload_path = write_share_payload(args.output_root, top_n=args.top_n)
        print(f"Share payload written to {payload_path}")

    for modality, result in results.items():
        print(
            f"Processed {len(result.full_table)} retrieval rows for {modality} "
            f"(threshold={result.high_threshold:.3f})."
        )


if __name__ == "__main__":
    main()

