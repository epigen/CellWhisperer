#!/usr/bin/env python
"""
Generate a single subplot grid of performance vs subsampling proportion across modality pairs.
Each axis is a simple lineplot where x is subsampling proportion [1, 8, 64, 512],
y is performance for a given metric, and two lines compare:
- Pair-only training (only the target modality pair data)
- With-bridge training (target pair data + the other modality pairs included)

Inputs via Snakemake params:
- snakemake.params.benchmarks_dir: base directory for aggregated results
- snakemake.params.ratios: list of subsampling ratios, e.g., [1, 8, 64, 256]

Output:
- snakemake.output.plot: path to save the grid plot PNG
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

benchmarks_dir = Path(snakemake.params.benchmarks_dir)
ratios = list(snakemake.params.ratios)

# Define modality pairs (columns)
modality_pairs = ["transcriptome-text", "transcriptome-image", "image-text"]

# Map test_dataset row names for retrieval CSVs
test_dataset_map = {
    "transcriptome-text": "cellxgene_census__archs4_geo",
    "transcriptome-image": "hest1k",
    "image-text": "quilt1m",
}

# Metrics per modality pair (rows). Keep concise small set matching existing subset_performance script.
metrics_by_pair = {
    "transcriptome-text": [
        "test_retrieval/left_right/rocauc_macroAvg",
        "test_retrieval/right_left/rocauc_macroAvg",
        "test_retrieval/left_right/recall_at_50_macroAvg",
        "test_retrieval/right_left/recall_at_50_macroAvg",
        "valfn_zshot_TabSap_cell_lvl/f1_macroAvg",
        "valfn_zshot_TabSap_cell_lvl/rocauc_macroAvg",
    ],
    "transcriptome-image": [
        "hest/overall_performance",
    ],
    "image-text": [
        "musk/pannuke_macro_avg_f1",
        "musk/skin_macro_avg_f1",
    ],
}

# Helper: build dataset combo name for given pair, ratio and bridge inclusion


def build_combo(modality_pair: str, ratio: int, include_bridge: bool) -> str:
    if modality_pair == "transcriptome-text":
        if ratio == 1:
            return (
                "cellxgene_census__archs4_geo__hest1k__quilt1m"
                if include_bridge
                else "cellxgene_census__archs4_geo"
            )
        suffix = f"{ratio}thsub"
        return (
            f"cellxgene_census_{suffix}__archs4_geo_{suffix}__hest1k__quilt1m"
            if include_bridge
            else f"cellxgene_census_{suffix}__archs4_geo_{suffix}"
        )
    elif modality_pair == "transcriptome-image":
        if ratio == 1:
            return (
                "cellxgene_census__archs4_geo__hest1k__quilt1m"
                if include_bridge
                else "hest1k"
            )
        suffix = f"{ratio}thsub"
        return (
            f"cellxgene_census__archs4_geo__hest1k_{suffix}__quilt1m"
            if include_bridge
            else f"hest1k_{suffix}"
        )
    elif modality_pair == "image-text":
        if ratio == 1:
            return (
                "cellxgene_census__archs4_geo__hest1k__quilt1m"
                if include_bridge
                else "quilt1m"
            )
        suffix = f"{ratio}thsub"
        return (
            f"cellxgene_census__archs4_geo__hest1k__quilt1m_{suffix}"
            if include_bridge
            else f"quilt1m_{suffix}"
        )
    else:
        raise ValueError(f"Unknown modality_pair: {modality_pair}")


# Helper: extract metrics for a combo and pair


def extract_metrics_for_combo(modality_pair: str, combo: str) -> dict:
    out = {}
    # Retrieval metrics (present for all combos)
    ret_path = benchmarks_dir / "retrieval" / combo / "aggregated_retrieval.csv"
    df_ret = pd.read_csv(ret_path, index_col=0)
    ret_row = df_ret.loc[test_dataset_map[modality_pair]]

    # CW evals (present for all combos)
    cw_path = benchmarks_dir / "retrieval" / combo / "aggregated_cwevals.csv"
    df_cw = pd.read_csv(cw_path, index_col=0, header=None).squeeze("columns")

    if modality_pair == "transcriptome-text":
        for direction in ["left_right", "right_left"]:
            for metric in ["rocauc_macroAvg", "recall_at_50_macroAvg"]:
                key = f"test_retrieval/{direction}/{metric}"
                out[key] = float(ret_row[key])
        for cw_key in [
            "valfn_zshot_TabSap_cell_lvl/f1_macroAvg",
            "valfn_zshot_TabSap_cell_lvl/rocauc_macroAvg",
        ]:
            out[cw_key] = float(df_cw.loc[cw_key])

    elif modality_pair == "transcriptome-image":
        # HEST benchmark overall performance
        hest_path = benchmarks_dir / "hest" / combo / "aggregated_results.json"
        with open(hest_path, "r") as f:
            hest_json = json.load(f)
        out["hest/overall_performance"] = float(
            hest_json["overall_performance"]
        )  # matches metrics_by_pair key

    elif modality_pair == "image-text":
        musk_path = benchmarks_dir / "musk" / combo / "performance_summary.json"
        with open(musk_path, "r") as f:
            musk_json = json.load(f)
        out["musk/pannuke_macro_avg_f1"] = float(
            musk_json["task_summaries"]["zeroshot_classification"]["macro_avg_f1"][
                "pannuke"
            ]
        )  # macro avg f1
        out["musk/skin_macro_avg_f1"] = float(
            musk_json["task_summaries"]["zeroshot_classification"]["macro_avg_f1"][
                "skin"
            ]
        )  # macro avg f1

    return out


# Determine grid size: top-aligned per column
ncols = len(modality_pairs)
nrows = max(len(metrics_by_pair[mp]) for mp in modality_pairs)
fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(3.2 * ncols, 2.2 * nrows), sharex=False
)


# Define line styles/colors
pair_only_color = "#4e79a7"
with_bridge_color = "#e15759"

# Plot grid, one column at a time, top-aligned
for col, mp in enumerate(modality_pairs):
    metrics = metrics_by_pair[mp]

    # Precompute metrics across ratios for both lines
    pair_only_values_by_metric = {m: [] for m in metrics}
    with_bridge_values_by_metric = {m: [] for m in metrics}

    for ratio in ratios:
        combo_pair_only = build_combo(mp, ratio, include_bridge=False)
        combo_with_bridge = build_combo(mp, ratio, include_bridge=True)
        vals_pair_only = extract_metrics_for_combo(mp, combo_pair_only)
        vals_with_bridge = extract_metrics_for_combo(mp, combo_with_bridge)
        for m in metrics:
            pair_only_values_by_metric[m].append(vals_pair_only[m])
            with_bridge_values_by_metric[m].append(vals_with_bridge[m])

    # Place plots at the top, fill remaining rows with empty axes
    for row in range(nrows):
        ax = axs[row, col] if nrows > 1 else axs[col]
        if row < len(metrics):
            metric = metrics[row]
            ax.plot(ratios, pair_only_values_by_metric[metric], marker="o", color=pair_only_color, label="pair-only" if (row == 0 and col == 0) else None)
            ax.plot(ratios, with_bridge_values_by_metric[metric], marker="o", color=with_bridge_color, label="with-bridge" if (row == 0 and col == 0) else None)
            ax.set_ylim(bottom=0)
            ax.set_xscale("log")
            ax.set_xticks(ratios)
            ax.set_xticklabels(["1", "1/8", "1/64", "1/512"])  # label fractions
            ax.set_xlim(ratios[0], ratios[-1])
            ax.set_title(metric)
        else:
            ax.axis("off")
        ax.set_ylabel("")


# Single legend
handles, labels = (
    axs[0, 0].get_legend_handles_labels()
    if nrows > 1
    else axs[0].get_legend_handles_labels()
)
if handles:
    fig.legend(handles, labels, loc="lower center", ncol=2)
    plt.subplots_adjust(bottom=0.08)

plt.tight_layout()
plt.savefig(snakemake.output.plot)
