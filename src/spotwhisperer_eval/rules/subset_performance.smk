# Subset performance comparison for a given modality pair
# NOTE: Depends only on existing aggregated outputs; does not retrain models

# This rule collects retrieval and benchmark results for five model setups
# per modality pair:
# 1) Bimodal matching (trained on the given modality pair)
# 2) 1/8th of the given modality pair
# 3) Bimodal bridge (trained on the other modality pair(s))
# 4) Bimodal bridge + 1/8th of the given modality pair
# 5) Trimodal (all modality pairs, full data)

from pathlib import Path
import shutil

# Map a modality pair to its corresponding dataset combinations

def models_for_pair(modality_pair, subratio):
    suffix = f"{subratio}thsub"
    if modality_pair == "transcriptome-text":
        base = "cellxgene_census__archs4_geo"
        sub = f"cellxgene_census_{suffix}__archs4_geo_{suffix}"
        bridge = "hest1k__quilt1m"
        bridge_plus_sub = f"cellxgene_census_{suffix}__archs4_geo_{suffix}__hest1k__quilt1m"
    elif modality_pair == "transcriptome-image":
        base = "hest1k"
        sub = f"hest1k_{suffix}"
        bridge = "cellxgene_census__archs4_geo__quilt1m"
        bridge_plus_sub = f"cellxgene_census__archs4_geo__hest1k_{suffix}__quilt1m"
    elif modality_pair == "image-text":
        base = "quilt1m"
        sub = f"quilt1m_{suffix}"
        bridge = "cellxgene_census__archs4_geo__hest1k"
        bridge_plus_sub = f"cellxgene_census__archs4_geo__hest1k__quilt1m_{suffix}"
    else:
        raise ValueError(f"Unknown modality_pair: {modality_pair}")

    trimodal = "cellxgene_census__archs4_geo__hest1k__quilt1m"
    return [base, sub, bridge, bridge_plus_sub, trimodal]

# Later ratios to consider (comment): 4, 16

PLOT_TRIMODAL_ALL_SUBSET = False

rule subset_performance:
    """
    Collect retrieval (aggregated_retrieval.csv, aggregated_cwevals.csv) and HEST benchmark
    results (aggregated_results.json) for five model setups corresponding to a modality pair.
    Outputs are copied into a consolidated directory under benchmarks/subset_performance.
    """
    input:
        retrieval_files=lambda wildcards: expand(
            BENCHMARKS_DIR / "retrieval" / "{combo}" / "aggregated_retrieval.csv",
            combo=models_for_pair(wildcards.modality_pair, wildcards.subratio)
        ),
        cwevals_files=lambda wildcards: expand(
            BENCHMARKS_DIR / "retrieval" / "{combo}" / "aggregated_cwevals.csv",
            combo=models_for_pair(wildcards.modality_pair, wildcards.subratio)
        ),
        hest_files=lambda wildcards: (
            [] if wildcards.modality_pair != "transcriptome-image" else expand(
                BENCHMARKS_DIR / "hest" / "{combo}" / "aggregated_results.json",
                combo=models_for_pair(wildcards.modality_pair, wildcards.subratio),
            )
        ),
        pathocell_files=lambda wildcards: expand(
            BENCHMARKS_DIR / "pathocell" / "{combo}" / "performance_summary.json",
            combo=models_for_pair(wildcards.modality_pair, wildcards.subratio),
            allow_missing=True,
        ),
        musk_files=lambda wildcards: (
            [] if wildcards.modality_pair != "image-text" else expand(
                BENCHMARKS_DIR / "musk" / "{combo}" / "performance_summary.json",
                # disable pannuke combos by filtering to combos including 'skin' implicitly via summary
                combo=models_for_pair(wildcards.modality_pair, wildcards.subratio),
            )
        )
    output:
        summary=BENCHMARKS_DIR / "subset_performance" / "{modality_pair}" / "{subratio}" / "summary.csv",
        plot=BENCHMARKS_DIR / "subset_performance" / "{modality_pair}" / "{subratio}" / "comparison.png"

    params:
        modality_pair=lambda wildcards: wildcards.modality_pair,
        subratio=lambda wildcards: int(wildcards.subratio)
    conda:
        "cellwhisperer"
    resources:
        mem_mb=5000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/aggregate_subset_performance.py"

rule subset_all:
    """
    Aggregate subset_performance across all three modality pairs for a given ratio
    by building their plots and writing a combined manifest.
    """
    input:
        text_plot=BENCHMARKS_DIR / "subset_performance" / "transcriptome-text" / "{subratio}" / "comparison.png",
        image_transcriptome_plot=BENCHMARKS_DIR / "subset_performance" / "transcriptome-image" / "{subratio}" / "comparison.png",
        image_text_plot=BENCHMARKS_DIR / "subset_performance" / "image-text" / "{subratio}" / "comparison.png"
    output:
        combined_manifest=BENCHMARKS_DIR / "subset_performance" / "{subratio}" / "combined_manifest.txt"

    conda:
        "cellwhisperer"
    resources:
        mem_mb=2000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/aggregate_subset_all.py"

# New plot: single grid of metric trends across subsampling ratios and modality pairs

def combos_for_grid(ratios):
    combos = []
    for r in ratios:
        suffix = f"{r}thsub"
        # transcriptome-text
        combos.append("cellxgene_census__archs4_geo" if r == 1 else f"cellxgene_census_{suffix}__archs4_geo_{suffix}")
        combos.append("cellxgene_census__archs4_geo__hest1k__quilt1m" if r == 1 else f"cellxgene_census_{suffix}__archs4_geo_{suffix}__hest1k__quilt1m")
        # transcriptome-image
        combos.append("hest1k" if r == 1 else f"hest1k_{suffix}")
        combos.append("cellxgene_census__archs4_geo__hest1k__quilt1m" if r == 1 else f"cellxgene_census__archs4_geo__hest1k_{suffix}__quilt1m")
        # image-text
        combos.append("quilt1m" if r == 1 else f"quilt1m_{suffix}")
        combos.append("cellxgene_census__archs4_geo__hest1k__quilt1m" if r == 1 else f"cellxgene_census__archs4_geo__hest1k__quilt1m_{suffix}")
    return combos

rule subset_performance_trend_grid:
    """
    Build a grid of lineplots showing performance vs subsampling proportion
    for each metric and modality pair, comparing pair-only vs with-bridge training.
    """
    input:
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        retrieval_files=lambda wildcards: expand(  # can probably be dropped
            BENCHMARKS_DIR / "retrieval" / "{combo}" / "aggregated_retrieval.csv",
            combo=combos_for_grid(SUBSAMPLING_RATIOS) + (
                [
                    (f"cellxgene_census_{r}thsub__archs4_geo_{r}thsub__hest1k_{r}thsub__quilt1m_{r}thsub")
                    for r in SUBSAMPLING_RATIOS if r != 1
                ] if PLOT_TRIMODAL_ALL_SUBSET else []
            ) + [
                # Baseline combos used by the plotting script
                "hest1k__quilt1m",
                "cellxgene_census__archs4_geo__quilt1m",
                "cellxgene_census__archs4_geo__hest1k",
                # "random",
            ],
            allow_missing=True,
        ),
        cwevals_files=lambda wildcards: expand(
            BENCHMARKS_DIR / "retrieval" / "{combo}" / "aggregated_cwevals.csv",
            combo=combos_for_grid(SUBSAMPLING_RATIOS) + (
                [
                    (f"cellxgene_census_{r}thsub__archs4_geo_{r}thsub__hest1k_{r}thsub__quilt1m_{r}thsub")
                    for r in SUBSAMPLING_RATIOS if r != 1
                ] if PLOT_TRIMODAL_ALL_SUBSET else []
            ) + [
                # Baseline combos used by the plotting script
                "hest1k__quilt1m",
                "cellxgene_census__archs4_geo__quilt1m",
                "cellxgene_census__archs4_geo__hest1k",
                # "random",
            ],
            allow_missing=True,
        ),
        hest_files=lambda wildcards: expand(
            BENCHMARKS_DIR / "hest" / "{combo}" / "aggregated_results.json",
            combo=[
                ("cellxgene_census__archs4_geo__hest1k__quilt1m" if r == 1 else f"cellxgene_census__archs4_geo__hest1k_{r}thsub__quilt1m")
                for r in SUBSAMPLING_RATIOS
            ] + [
                ("hest1k" if r == 1 else f"hest1k_{r}thsub")
                for r in SUBSAMPLING_RATIOS
            ] + (
                [
                    (f"cellxgene_census_{r}thsub__archs4_geo_{r}thsub__hest1k_{r}thsub__quilt1m_{r}thsub")
                    for r in SUBSAMPLING_RATIOS if r != 1
                ] if PLOT_TRIMODAL_ALL_SUBSET else []
            ) + [
                # Bimodal bridge baseline for transcriptome-image column
                "cellxgene_census__archs4_geo__quilt1m",
                # "random",
            ],
            allow_missing=True,
        ),
        # PathoCellBench inputs intentionally excluded from the trend grid
        # pathocell_files=lambda wildcards: expand(
        #     PROJECT_DIR / "results/pathocell_evaluation" / "spotwhisperer_{combo}" / "summary/patch_metrics_from_scores_aggregated.json",
        #     combo=[
        #         # image-text: pair-only
        #         ("quilt1m" if r == 1 else f"quilt1m_{r}thsub")
        #         for r in SUBSAMPLING_RATIOS
        #     ] + [
        #         # image-text: with-bridge
        #         ("cellxgene_census__archs4_geo__hest1k__quilt1m" if r == 1 else f"cellxgene_census__archs4_geo__hest1k__quilt1m_{r}thsub")
        #         for r in SUBSAMPLING_RATIOS
        #     ] + (
        #         [
        #             # trimodal-all-subset (optional)
        #             (f"cellxgene_census_{r}thsub__archs4_geo_{r}thsub__hest1k_{r}thsub__quilt1m_{r}thsub")
        #             for r in SUBSAMPLING_RATIOS if r != 1
        #         ] if PLOT_TRIMODAL_ALL_SUBSET else []
        #     ) + [
        #         # image-text baseline (bimodal bridge)
        #         "cellxgene_census__archs4_geo__hest1k"
        #     ],
        #     allow_missing=True,
        # ),
        musk_files=lambda wildcards: expand(
            BENCHMARKS_DIR / "musk" / "{combo}" / "performance_summary.json",
            combo=[
                ("cellxgene_census__archs4_geo__hest1k__quilt1m" if r == 1 else f"cellxgene_census__archs4_geo__hest1k__quilt1m_{r}thsub")
                for r in SUBSAMPLING_RATIOS
            ] + [
                ("quilt1m" if r == 1 else f"quilt1m_{r}thsub")
                for r in SUBSAMPLING_RATIOS
            ] + (
                [
                    (f"cellxgene_census_{r}thsub__archs4_geo_{r}thsub__hest1k_{r}thsub__quilt1m_{r}thsub")
                    for r in SUBSAMPLING_RATIOS if r != 1
                ] if PLOT_TRIMODAL_ALL_SUBSET else []
            ) + [
                # Bimodal bridge baseline for image-text column
                "cellxgene_census__archs4_geo__hest1k",
                # "random",
            ],
            allow_missing=True,
        )
    output:
        plot=BENCHMARKS_DIR / "subset_performance" / "trend_grid.svg"
    params:
        benchmarks_dir=BENCHMARKS_DIR,
        ratios=SUBSAMPLING_RATIOS,
        plot_trimodal_all_subset=PLOT_TRIMODAL_ALL_SUBSET
    conda:
        "cellwhisperer"
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=2"
    script:
        "../scripts/plot_subset_trends.py"
