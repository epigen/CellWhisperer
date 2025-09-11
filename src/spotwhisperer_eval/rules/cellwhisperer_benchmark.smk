# CellWhisperer zero-shot benchmark evaluation
# Reimplements zero-shot predictions from figures/fig2_embedding_validations.smk

include: "../../shared/config.smk"
include: "../../shared/rules/dataset_processing.smk"

# CellWhisperer benchmark datasets (from the original validation)
CELLWHISPERER_BENCHMARK_DATASETS = [
    "tabula_sapiens", 
    "tabula_sapiens_well_studied_celltypes",
    "aida",
    "pancreas"
]

CELLWHISPERER_METADATA_COLS = {
    "tabula_sapiens": ["celltype"],
    "tabula_sapiens_well_studied_celltypes": ["celltype"], 
    "aida": ["celltype"],
    "pancreas": ["celltype"]
}

# Results paths
CELLWHISPERER_BENCHMARK_RESULTS = PROJECT_DIR / "results/spotwhisperer_eval/benchmarks/cellwhisperer"
CELLWHISPERER_MODEL_RESULTS = CELLWHISPERER_BENCHMARK_RESULTS / "{model_name}"

rule cellwhisperer_zero_shot_prediction:
    """
    Zero-shot cell type prediction using trained SpotWhisperer model
    
    Reimplements the logic from plot_confusion_matrix_and_get_performance_metrics
    """
    input:
        processed_dataset=lambda wildcards: str(PROJECT_DIR / config["paths"]["model_processed_dataset"]).format(model=config["model_name_path_map"]["cellwhisperer"], dataset=wildcards.dataset),
        raw_read_count_table=lambda wildcards: str(PROJECT_DIR / config["paths"]["read_count_table"]).format(dataset=wildcards.dataset),
        model="{model_path}",
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        confusion_matrix_plot=CELLWHISPERER_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "confusionmatrix{normed}.pdf",
        confusion_matrix_table=CELLWHISPERER_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "confusionmatrix{normed}.xlsx",
        performance_metrics=CELLWHISPERER_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "performancemetrics{normed}.csv",
        performance_metrics_per_metadata=CELLWHISPERER_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "performance_metrics_permetadata{normed}.csv"
    params:
        normed=lambda wildcards: wildcards.normed == "normed",
        use_prefix_suffix_version=True
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
        notebook="../logs/cellwhisperer_zero_shot_prediction_{model_name}_{dataset}_{metadata_col}_{normed}.ipynb"
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
            metadata_col=["celltype"],
            normed=["raw"],
            model_name=wildcards.model_name
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
        notebook="../logs/cellwhisperer_benchmark_summary_{model_name}.ipynb"
    notebook:
        "../notebooks/cellwhisperer_benchmark_summary.py.ipynb"

rule cellwhisperer_benchmark_all:
    """
    Run complete CellWhisperer benchmark for a given model
    """
    input:
        # All confusion matrices and performance metrics
        lambda wildcards: expand(
            CELLWHISPERER_MODEL_RESULTS / "datasets" / "{dataset}" / "{metadata_col}" / "confusionmatrix{normed}.pdf",
            dataset=CELLWHISPERER_BENCHMARK_DATASETS,
            metadata_col=["celltype"],
            normed=["normed", "raw"],
            model_name=wildcards.model_name
        ),
        # Summary results
        CELLWHISPERER_MODEL_RESULTS / "benchmark_summary.csv"