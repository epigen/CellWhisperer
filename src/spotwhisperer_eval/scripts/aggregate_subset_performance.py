#!/usr/bin/env python
"""
Build a per-modality-pair performance summary and plot.
Small metric set included; comprehensive options commented for future expansion.

Inputs (Snakemake):
- snakemake.input.retrieval_files: list of aggregated_retrieval.csv for 5 combos
- snakemake.input.cwevals_files: list of aggregated_cwevals.csv for 5 combos
- snakemake.input.hest_files: aggregated_results.json for 5 combos (only for transcriptome-image)
- snakemake.output.summary: summary CSV path
- snakemake.input.pathocell_files: performance_summary.json for 5 combos (PathoCellBench)
- snakemake.output.plot: comparison PNG path
- snakemake.params.modality_pair: pair string
"""
from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt

modality_pair = snakemake.params.modality_pair
subratio = snakemake.params.subratio

# Model labels in order (dynamic subratio labels)
model_labels = [
    "bimodal_matching",
    f"sub_1/{subratio}",
    "bimodal_bridge",
    f"bridge_plus_1/{subratio}",
    "trimodal",
]

rows = []

# Helper: extract small metric set per modality
for i in range(len(snakemake.input.retrieval_files)):
    combo = Path(snakemake.input.retrieval_files[i]).parent.parent.name
    df_ret = pd.read_csv(snakemake.input.retrieval_files[i], index_col=0)
    df_cw = pd.read_csv(
        snakemake.input.cwevals_files[i], index_col=0, header=None
    ).squeeze("columns")

    label = model_labels[i]

    # Select the matching test_dataset row for retrieval
    test_dataset_map = {
        "transcriptome-text": "cellxgene_census__archs4_geo",
        "transcriptome-image": "hest1k",
        "image-text": "quilt1m",
    }
    ret_row = df_ret.loc[test_dataset_map[modality_pair]]

    if modality_pair == "transcriptome-text":
        # Small set: retrieval rocauc and recall@50 in both directions; TabSap CW f1/rocauc
        for direction in ["left_right", "right_left"]:
            for metric in ["rocauc_macroAvg", "recall_at_50_macroAvg"]:
                key = f"test_retrieval/{direction}/{metric}"
                if key in ret_row.index:
                    rows.append(
                        {"metric": key, "model": label, "value": float(ret_row[key])}
                    )
        # CW metrics (Tabula Sapiens)
        for cw_key in [
            "valfn_zshot_TabSap_cell_lvl/f1_macroAvg",
            "valfn_zshot_TabSap_cell_lvl/rocauc_macroAvg",
        ]:
            if cw_key in df_cw.index:
                rows.append(
                    {
                        "metric": cw_key,
                        "model": label,
                        "value": float(df_cw.loc[cw_key]),
                    }
                )
        # Comprehensive options (commented): include WellStudied, immgen, human_disease, pancreas
        # for cw_key in [
        #     "valfn_zshot_TabSapWellStudied_cell_lvl/f1_macroAvg",
        #     "valfn_zshot_TabSapWellStudied_cell_lvl/rocauc_macroAvg",
        #     "valfn_immgen_deduplicated/text_as_classes_f1_macroAvg",
        #     "valfn_immgen_deduplicated/text_as_classes_rocauc_macroAvg",
        #     "valfn_human_disease_strictly_deduplicated_dmis-lab_biobert-v1.1_CLS_pooling/text_as_classes_f1_macroAvg",
        #     "valfn_human_disease_strictly_deduplicated_dmis-lab_biobert-v1.1_CLS_pooling/text_as_classes_rocauc_macroAvg",
        #     "valfn_pancreas/text_as_classes_f1_macroAvg",
        #     "valfn_pancreas/text_as_classes_rocauc_macroAvg",
        # ]:
        #     if cw_key in df_cw.index:
        #         rows.append({"metric": cw_key, "model": label, "value": float(df_cw.loc[cw_key])})

    elif modality_pair == "transcriptome-image":
        # Small set: HEST overall metric from aggregated_results.json
        with open(snakemake.input.hest_files[i], "r") as f:
            hest_json = json.load(f)
        rows.append(
            {
                "metric": "hest/overall_performance",
                "model": label,
                "value": float(hest_json["overall_performance"]),
            }
        )
        # PathoCellBench: include zero-shot classification (F1 macro average) from performance_summary.json
        with open(snakemake.input.pathocell_files[i], "r") as f:
            patho = json.load(f)
        rows.append(
            {
                "metric": "pathocell/zero_shot_classification",
                "model": label,
                "value": float(patho["pathocell_zero_shot_classification"]),
            }
        )

    elif modality_pair == "image-text":
        # Small set: MUSK pannuke/skin ROC-AUC (use ROC-AUC for subset plot consistency)
        with open(snakemake.input.musk_files[i], "r") as f:
            musk_json = json.load(f)
        for dataset in ["pannuke", "skin"]:
            rows.append(
                {
                    "metric": f"musk/{dataset}_macro_avg_rocauc",
                    "model": label,
                    "value": float(
                        musk_json["task_summaries"]["zeroshot_classification"][
                            "macro_avg_rocauc"
                        ][dataset]
                    ),
                }
            )

        # Comprehensive options: add retrieval and few-shot metrics
        # for dataset in ["pathmmu_retrieval", "unitopatho_retrieval"]:
        #     for m in ["rocauc", "recall@1","recall@10","recall@50"]:
        #         key = f"{dataset}_{m}"
        #         if key in musk_json:
        #             rows.append({"metric": f"musk/{key}", "model": label, "value": float(musk_json[key])})

# Build summary and write
summary_df = pd.DataFrame(rows)
summary_df.to_csv(snakemake.output.summary, index=False)

# Plot: grouped bars per metric
plt.figure(figsize=(12, 6))
# Order metrics by appearance
metrics_order = list(dict.fromkeys(summary_df["metric"].tolist()))
# Build grouped bars
x_positions = range(len(metrics_order))
bar_width = 0.15
colors = {
    "bimodal_matching": "#4e79a7",
    f"sub_1/{subratio}": "#f28e2b",
    "bimodal_bridge": "#e15759",
    f"bridge_plus_1/{subratio}": "#76b7b2",
    "trimodal": "#59a14f",
}
for j, label in enumerate(model_labels):
    values = []
    for metric in metrics_order:
        # value for label+metric
        subset = summary_df[
            (summary_df["model"] == label) & (summary_df["metric"] == metric)
        ]
        values.append(float(subset["value"].iloc[0]) if len(subset) else float("nan"))
    plt.bar(
        [x + j * bar_width for x in x_positions],
        values,
        width=bar_width,
        label=label,
        color=colors[label],
    )

plt.xticks(
    [x + 2 * bar_width for x in x_positions], metrics_order, rotation=45, ha="right"
)
plt.ylabel("Performance")
plt.title(f"Subset Performance Comparison ({modality_pair})")
plt.legend()
plt.tight_layout()
plt.savefig(snakemake.output.plot)
