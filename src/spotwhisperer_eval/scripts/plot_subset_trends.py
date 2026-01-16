#!/usr/bin/env python
"""
Generate a single subplot grid of performance vs subsampling proportion across modality pairs.
Each axis is a simple lineplot where x is subsampling proportion [1, 8, 64, 512],
y is performance for a given metric, and three lines compare:
- Pair-only training (only the target modality pair data)
- With-bridge training (target pair data + the other modality pairs included)
- Trimodal-all-subset (trimodal model with all datasets subsetted)

Inputs via Snakemake params:
- snakemake.params.benchmarks_dir: base directory for aggregated results
- snakemake.params.ratios: list of subsampling ratios, e.g., [1, 8, 64, 256]
- snakemake.params.plot_trimodal_all_subset: bool toggle to include third line

Output:
- snakemake.output.plot: path to save the grid plot PNG
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

benchmarks_dir = Path(snakemake.params.benchmarks_dir)
ratios = list(snakemake.params.ratios)
plot_trimodal_all_subset = bool(snakemake.params.plot_trimodal_all_subset)

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
        "valfn_zshot_TabSap_cell_lvl/f1_macroAvg",
        "valfn_zshot_TabSap_cell_lvl/rocauc_macroAvg",
        "valfn_zshot_HumanDisease_disease_subtype/rocauc_macroAvg",
    ],
    "transcriptome-image": [
        "hest/overall_performance",
    ],
    "image-text": [
        "musk/pannuke_macro_avg_rocauc",
        "musk/skin_macro_avg_rocauc",
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


def build_trimodal_all_subset_combo(ratio: int) -> str:
    if ratio == 1:
        return "cellxgene_census__archs4_geo__hest1k__quilt1m"
    suffix = f"{ratio}thsub"
    return f"cellxgene_census_{suffix}__archs4_geo_{suffix}__hest1k_{suffix}__quilt1m_{suffix}"


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
            "valfn_zshot_HumanDisease_disease_subtype/rocauc_macroAvg",
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
        out["musk/pannuke_macro_avg_rocauc"] = float(
            musk_json["task_summaries"]["zeroshot_classification"]["macro_avg_rocauc"][
                "pannuke"
            ]
        )  # macro avg rocauc
        out["musk/skin_macro_avg_rocauc"] = float(
            musk_json["task_summaries"]["zeroshot_classification"]["macro_avg_rocauc"][
                "skin"
            ]
        )  # macro avg rocauc

    return out


# Baseline combos per modality pair


def get_baseline_combo(modality_pair: str, model: str) -> str:
    if model == "trimodal":
        return "cellxgene_census__archs4_geo__hest1k__quilt1m"
    elif model == "bimodal_bridge":
        if modality_pair == "transcriptome-text":
            return "hest1k__quilt1m"
        elif modality_pair == "transcriptome-image":
            return "cellxgene_census__archs4_geo__quilt1m"
        elif modality_pair == "image-text":
            return "cellxgene_census__archs4_geo__hest1k"
        else:
            raise ValueError(f"Unknown modality_pair: {modality_pair}")
    else:
        raise ValueError(f"Unknown baseline model: {model}")


# Determine grid size: top-aligned per column
ncols = len(modality_pairs)
nrows = max(len(metrics_by_pair[mp]) for mp in modality_pairs)
fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(3.2 * ncols, 2.2 * nrows), sharex=False
)


# Define line styles/colors
pair_only_color = "#4e79a7"
with_bridge_color = "#e15759"
trimodal_all_color = "#59a14f"

# Plot grid, one column at a time, top-aligned
for col, mp in enumerate(modality_pairs):
    metrics = metrics_by_pair[mp]

    # Precompute metrics across ratios for required lines
    pair_only_values_by_metric = {m: [] for m in metrics}
    with_bridge_values_by_metric = {m: [] for m in metrics}
    trimodal_all_values_by_metric = {m: [] for m in metrics}

    for ratio in ratios:
        combo_pair_only = build_combo(mp, ratio, include_bridge=False)
        combo_with_bridge = build_combo(mp, ratio, include_bridge=True)
        vals_pair_only = extract_metrics_for_combo(mp, combo_pair_only)
        vals_with_bridge = extract_metrics_for_combo(mp, combo_with_bridge)
        for m in metrics:
            pair_only_values_by_metric[m].append(vals_pair_only[m])
            with_bridge_values_by_metric[m].append(vals_with_bridge[m])
        if plot_trimodal_all_subset:
            combo_trimodal_all = build_trimodal_all_subset_combo(ratio)
            vals_trimodal_all = extract_metrics_for_combo(mp, combo_trimodal_all)
            for m in metrics:
                trimodal_all_values_by_metric[m].append(vals_trimodal_all[m])

    # Baseline combos for horizontal lines
    trimodal_combo = get_baseline_combo(mp, "trimodal")
    bimodal_bridge_combo = get_baseline_combo(mp, "bimodal_bridge")
    trimodal_vals = extract_metrics_for_combo(mp, trimodal_combo)
    bimodal_bridge_vals = extract_metrics_for_combo(mp, bimodal_bridge_combo)

    # Place plots at the top, fill remaining rows with empty axes
    for row in range(nrows):
        ax = axs[row, col] if nrows > 1 else axs[col]
        if row < len(metrics):
            metric = metrics[row]
            ax.plot(
                ratios,
                pair_only_values_by_metric[metric],
                marker="o",
                color=pair_only_color,
                label="pair-only" if (row == 0 and col == 0) else None,
            )
            ax.plot(
                ratios,
                with_bridge_values_by_metric[metric],
                marker="o",
                color=with_bridge_color,
                label="with-bridge" if (row == 0 and col == 0) else None,
            )
            if plot_trimodal_all_subset:
                ax.plot(
                    ratios,
                    trimodal_all_values_by_metric[metric],
                    marker="o",
                    color=trimodal_all_color,
                    label="trimodal-all-subset" if (row == 0 and col == 0) else None,
                )
            ax.set_ylim(bottom=0)
            ax.set_xscale("log")
            ax.set_xticks(ratios)
            ax.set_xticklabels(["1", "1/8", "1/64", "1/512"])  # label fractions
            ax.set_xlim(ratios[0], ratios[-1])
            ax.set_title(metric)
            # Add horizontal baselines: trimodal and bimodal_bridge (constant across ratios)
            trimodal_baseline = float(trimodal_vals[metric])
            bimodal_bridge_baseline = float(bimodal_bridge_vals[metric])
            ax.axhline(
                trimodal_baseline,
                color="#333333",
                linestyle="--",
                linewidth=1.2,
                label="trimodal baseline" if (row == 0 and col == 0) else None,
            )
            ax.axhline(
                bimodal_bridge_baseline,
                color="#777777",
                linestyle="--",
                linewidth=1.2,
                label="bimodal_bridge baseline" if (row == 0 and col == 0) else None,
            )

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
    legend_ncol = 3 if plot_trimodal_all_subset else 2
    fig.legend(handles, labels, loc="lower center", ncol=legend_ncol)
    plt.subplots_adjust(bottom=0.1)

plt.tight_layout()
out_path = Path(snakemake.output.plot)
plt.savefig(out_path)
plt.savefig(out_path.with_suffix(".svg"))
plt.savefig(out_path.with_suffix(".pdf"))
