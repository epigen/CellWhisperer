# CellWhisperer zero-shot benchmark evaluation
# Reimplements zero-shot predictions from figures/fig2_embedding_validations.smk

# CellWhisperer benchmark datasets (from the original validation)
CELLWHISPERER_BENCHMARK_DATASETS = [
    "tabula_sapiens",
    "tabula_sapiens_well_studied_celltypes",
    # "aida",
    "pancreas",
    "human_disease",
    "immgen"
]

# Results paths
CELLWHISPERER_BENCHMARK_RESULTS = PROJECT_DIR / "results/spotwhisperer_eval/benchmarks/cellwhisperer"
CELLWHISPERER_MODEL_RESULTS = CELLWHISPERER_BENCHMARK_RESULTS / "{model}"

rule cellwhisperer_zero_shot_prediction:
    """
    NOTE: This might be obsolete due to `cellwhisperer test` command (in main Snakefile)
    Zero-shot cell type prediction on benchmark datasets using a trained model.
    """
    input:
        processed_dataset=PROJECT_DIR / config["paths"]["model_processed_dataset"],
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        performance_metrics=CELLWHISPERER_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "performancemetrics{normed}.csv",
        performance_metrics_per_metadata=CELLWHISPERER_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "performance_metrics_permetadata{normed}.csv"
    params:
        normed=lambda wildcards: wildcards.normed == "normed",
        use_prefix_suffix_version=True,
        average_mode=None  # TODO try alternative: "embeddings"
    wildcard_constraints:
        dataset="|".join(CELLWHISPERER_BENCHMARK_DATASETS),
        metadata_col="celltype",
        normed="normed|raw"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=230000,
        slurm=slurm_gres()
    log:
        notebook="../logs/cellwhisperer_zero_shot_prediction_{model}_{dataset}_{metadata_col}_{normed}.ipynb"
    notebook:
        "../notebooks/cellwhisperer_zero_shot_prediction.py.ipynb"

rule cellwhisperer_benchmark_summary:
    """
    NOTE: This might be obsolete due to `cellwhisperer test` command (in main Snakefile)
    Summarize zero-shot benchmark performance across all datasets for a model.
    """
    input:
        performance_files=lambda wildcards: expand(
            CELLWHISPERER_MODEL_RESULTS / "datasets" / "{dataset}" / "{metadata_col}" / "performancemetrics{normed}.csv",
            dataset=CELLWHISPERER_BENCHMARK_DATASETS,
            metadata_col=["celltype"],  # TODO need to provide the appropriate metadata columns for each of the corresponding datasets (following metadata_cols_per_zero_shot_validation_dataset in config.yaml)
            normed=["raw"],
            model=wildcards.model
        )
    output:
        summary_csv=CELLWHISPERER_MODEL_RESULTS / "benchmark_summary.csv",
        summary_plot=CELLWHISPERER_MODEL_RESULTS / "benchmark_summary.png"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=100000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/cellwhisperer_benchmark_summary_{model}.ipynb"
    notebook:
        "../notebooks/cellwhisperer_benchmark_summary.py.ipynb"

rule cellwhisperer_per_class_analysis:
    """
    Per-class comparison of trimodal vs bimodal models on benchmark datasets.
    """
    input:
        # Results from trimodal and bimodal_matching models
        lambda wildcards: [
            CELLWHISPERER_BENCHMARK_RESULTS / "spotwhisperer_{}".format(combo) / "datasets" / dataset / metadata_col / "performance_metrics_permetadata{}.csv".format(normed)
            for combo in ["cellxgene_census__archs4_geo__hest1k__quilt1m",  # trimodal
                          "hest1k", "quilt1m", "cellxgene_census__archs4_geo"]  # bimodal matching options
            for dataset in CELLWHISPERER_BENCHMARK_DATASETS
            for metadata_col in ["celltype"]
            for normed in ["raw"]
        ]
    output:
        analysis=report(CELLWHISPERER_BENCHMARK_RESULTS / "comparison" / "per_class_analysis.csv", category="per_class_analysis", subcategory="transcriptome-text", labels={"Analysis": "CellWhisperer benchmark", "Format": "csv"}),
        plot=report(CELLWHISPERER_BENCHMARK_RESULTS / "comparison" / "per_class_analysis.pdf", category="per_class_analysis", subcategory="transcriptome-text", labels={"Analysis": "CellWhisperer benchmark", "Format": "plot"})
    params:
        datasets=CELLWHISPERER_BENCHMARK_DATASETS,
        metrics=["rocauc", "f1", "accuracy", "precision", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_50"]  # All available metrics
    conda:
        "cellwhisperer"
    resources:
        mem_mb=20000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/cellwhisperer_per_class_analysis.ipynb"
    notebook:
        "../notebooks/cellwhisperer_per_class_analysis.py.ipynb"

rule cellwhisperer_benchmark_all:
    """
    Run the CellWhisperer benchmark end-to-end for the selected model.
    """
    input:
        # Summary results
        expand(CELLWHISPERER_MODEL_RESULTS / "benchmark_summary.csv",
               model="spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m"),
        # Per-class analysis
        rules.cellwhisperer_per_class_analysis.output.analysis
