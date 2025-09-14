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
    Zero-shot cell type prediction using trained SpotWhisperer model

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
        mem_mb=400000,
        slurm=slurm_gres()
    log:
        notebook="../logs/cellwhisperer_zero_shot_prediction_{model}_{dataset}_{metadata_col}_{normed}.ipynb"
    notebook:
        "../notebooks/cellwhisperer_zero_shot_prediction.py.ipynb"

rule cellwhisperer_benchmark_summary:
    """
    Summarize CellWhisperer benchmark performance across all datasets
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

rule cellwhisperer_benchmark_all:
    """
    Run complete CellWhisperer benchmark for a given model
    """
    input:
        # Summary results
        expand(CELLWHISPERER_MODEL_RESULTS / "benchmark_summary.csv",
               model="spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m")
