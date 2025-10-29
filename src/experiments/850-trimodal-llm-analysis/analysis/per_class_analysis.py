#!/usr/bin/env python
"""Reproducible per-class analysis for trimodal vs bimodal models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "docs" / "per_class_analysis"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_image_text(data_dir: Path) -> pd.DataFrame:
    image_dir = data_dir / "image-text"
    per_class_path = image_dir / "per_class_analysis.csv"
    musk_path = image_dir / "musk_per_class_analysis_seed0.csv"

    if per_class_path.exists():
        df = pd.read_csv(per_class_path)
        df = df.assign(
            modality="image-text",
            dataset="quilt1m",
            class_label=df["orig_ids"],
            bimodal=df["bimodal_matching"],
            trimodal=df["trimodal"],
            improvement=df["improvement"],
            relative_improvement=df["relative_improvement"],
        )
    elif musk_path.exists():
        df = pd.read_csv(musk_path)
        df = df[df["metric_type"].str.lower() == "f1"].copy()
        df = df.assign(
            modality="image-text",
            dataset=df["dataset"],
            class_label=df["class_name"],
            bimodal=df["bimodal_matching"],
            trimodal=df["trimodal"],
            improvement=df["improvement"],
            relative_improvement=df["relative_improvement"],
        )
    else:
        raise FileNotFoundError(
            "No per-class CSV found for image-text modality; expected `per_class_analysis.csv` or `musk_per_class_analysis_seed0.csv`."
        )

    return df[["modality", "dataset", "class_label", "bimodal", "trimodal", "improvement", "relative_improvement"]]


def load_transcriptome_text(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "transcriptome-text" / "human_disease_per_class_analysis.csv")
    df = df.assign(
        modality="transcriptome-text",
        dataset=df["dataset"],
        class_label=df["class"],
        bimodal=df["bimodal_matching"],
        trimodal=df["trimodal"],
        improvement=df["improvement"],
        relative_improvement=df["relative_improvement"],
    )
    return df[["modality", "dataset", "class_label", "bimodal", "trimodal", "improvement", "relative_improvement"]]


def load_transcriptome_image(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "transcriptome-image" / "hest1k_per_class_analysis.csv")
    df = df.assign(
        modality="transcriptome-image",
        dataset=df["dataset"],
        class_label=df["class_type"],
        bimodal=df["bimodal_matching"],
        trimodal=df["trimodal"],
        improvement=df["improvement"],
        relative_improvement=df["relative_improvement"],
    )
    return df[["modality", "dataset", "class_label", "bimodal", "trimodal", "improvement", "relative_improvement"]]


def load_all(data_dir: Path) -> pd.DataFrame:
    loaders = [load_image_text, load_transcriptome_text, load_transcriptome_image]
    frames = [loader(data_dir) for loader in loaders]
    combined = pd.concat(frames, ignore_index=True)
    return combined


def compute_stats(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    stats = {}
    stats["overall"] = (
        df.groupby("modality")["improvement"].agg(["count", "mean", "median", "std"]).reset_index()
    )

    stats["top_improvements"] = (
        df.sort_values("improvement", ascending=False)
        .groupby("modality")
        .head(10)
        .reset_index(drop=True)
    )

    stats["worst_declines"] = (
        df.sort_values("improvement", ascending=True)
        .groupby("modality")
        .head(10)
        .reset_index(drop=True)
    )

    stats["relative_summary"] = (
        df[df["relative_improvement"].notnull()]
        .groupby("modality")["relative_improvement"].agg(["mean", "median", "std"])
        .reset_index()
    )

    return stats


def export_stats(stats: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in stats.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    # simple JSON summary for quick loading later
    summary_payload = {
        name: json.loads(table.to_json(orient="records")) for name, table in stats.items()
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))


def export_full_df(df: pd.DataFrame, output_dir: Path) -> None:
    df.to_csv(output_dir / "per_class_combined.csv", index=False)


def generate_plots(df: pd.DataFrame, stats: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    overall = stats["overall"].rename(columns={"mean": "mean_improvement"})
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=overall, x="modality", y="mean_improvement", hue="modality", dodge=False)
    plt.title("Mean Improvement by Modality")
    plt.ylabel("Absolute improvement (trimodal - bimodal)")
    plt.xlabel("Modality pair")
    if ax.get_legend() is not None:
        sns.move_legend(ax, "upper right")
    plt.tight_layout()
    plt.savefig(plot_dir / "mean_improvement_by_modality.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.kdeplot(data=df, x="improvement", hue="modality", common_norm=False, fill=True, alpha=0.3)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Distribution of Per-Class Improvements")
    plt.xlabel("Improvement (trimodal - bimodal)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(plot_dir / "improvement_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.stripplot(data=df, x="modality", y="improvement", alpha=0.2)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Per-Class Improvement Scatter by Modality")
    plt.ylabel("Improvement (trimodal - bimodal)")
    plt.xlabel("Modality pair")
    plt.tight_layout()
    plt.savefig(plot_dir / "improvement_scatter.png", dpi=200)
    plt.close()

    for modality, group in df.groupby("modality"):
        plt.figure(figsize=(6, 4))
        sns.histplot(group["improvement"], bins=40, kde=True, alpha=0.6)
        plt.axvline(0, color="black", linestyle="--", linewidth=1)
        plt.title(f"Improvement Histogram – {modality}")
        plt.xlabel("Improvement (trimodal - bimodal)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(plot_dir / f"histogram_{modality.replace(' ', '_')}.png", dpi=200)
        plt.close()

    top = (
        df.sort_values("improvement", ascending=False)
        .groupby("modality")
        .head(10)
        .reset_index(drop=True)
    )
    bottom = (
        df.sort_values("improvement", ascending=True)
        .groupby("modality")
        .head(10)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, x="improvement", y="class_label", hue="modality", dodge=False)
    plt.title("Top 10 Per-Modality Improvements")
    plt.xlabel("Improvement (trimodal - bimodal)")
    plt.ylabel("Class label")
    plt.tight_layout()
    plt.savefig(plot_dir / "top10_improvements.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=bottom, x="improvement", y="class_label", hue="modality", dodge=False)
    plt.title("Worst 10 Per-Modality Declines")
    plt.xlabel("Improvement (trimodal - bimodal)")
    plt.ylabel("Class label")
    plt.tight_layout()
    plt.savefig(plot_dir / "bottom10_declines.png", dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-class trimodal vs bimodal analyses.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing per-class CSV outputs (default: docs/per_class_analysis)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to store aggregated outputs (default: analysis/outputs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_all(args.data_dir)
    stats = compute_stats(df)
    export_full_df(df, args.output_dir)
    export_stats(stats, args.output_dir)
    generate_plots(df, stats, args.output_dir)
    print(f"Processed {len(df)} per-class entries across {df['modality'].nunique()} modalities.")
    for name, table in stats.items():
        print(f"Saved {name} with shape {table.shape}")


if __name__ == "__main__":
    main()
