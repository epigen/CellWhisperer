# CellWhisperer comprehensive benchmark evaluation using the centralized registry
# This provides the same comprehensive evaluation as fig2_embedding_validations 
# but integrates with the spotwhisperer_eval pipeline

# Import the registry at the top level for Snakemake
import sys
sys.path.append(str(PROJECT_DIR / "src"))

# Lazy import to avoid issues if the module isn't available
try:
    from cellwhisperer.validation.registry import ValidationRegistry
    COMPREHENSIVE_BENCHMARKS = ValidationRegistry.get_cellwhisperer_benchmarks()
    BENCHMARK_NAMES = ValidationRegistry.get_benchmark_names()
    DATASET_METADATA_PAIRS = ValidationRegistry.get_dataset_metadata_pairs()
except ImportError:
    # Fallback if registry isn't available
    COMPREHENSIVE_BENCHMARKS = []
    BENCHMARK_NAMES = []
    DATASET_METADATA_PAIRS = []

# Results paths
COMPREHENSIVE_BENCHMARK_RESULTS = BENCHMARKS_DIR / "cellwhisperer_comprehensive"
COMPREHENSIVE_MODEL_RESULTS = COMPREHENSIVE_BENCHMARK_RESULTS / "{model}"

rule cellwhisperer_comprehensive_zero_shot:
    """
    Comprehensive zero-shot evaluation using the centralized benchmark registry.
    This mirrors fig2_embedding_validations but integrates with spotwhisperer_eval.
    """
    input:
        processed_dataset=PROJECT_DIR / config["paths"]["model_processed_dataset"],
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        performance_metrics=COMPREHENSIVE_MODEL_RESULTS / "datasets" / "{dataset}" / "{metadata_col}" / "performance_metrics.csv", 
        performance_metrics_per_metadata=COMPREHENSIVE_MODEL_RESULTS / "datasets" / "{dataset}" / "{metadata_col}" / "performance_metrics_per_metadata.csv"
    params:
        # Pass the benchmark specification to the notebook
        benchmark_name=lambda wildcards: f"{wildcards.dataset}_{wildcards.metadata_col}",
        use_prefix_suffix_version=True,
    wildcard_constraints:
        dataset="|".join(set([pair[0] for pair in DATASET_METADATA_PAIRS])) if DATASET_METADATA_PAIRS else ".*",
        metadata_col="|".join(set([pair[1] for pair in DATASET_METADATA_PAIRS])) if DATASET_METADATA_PAIRS else ".*"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=350000,
        slurm=slurm_gres()
    log:
        notebook="../logs/cellwhisperer_comprehensive_zero_shot_{model}_{dataset}_{metadata_col}.ipynb" 
    notebook:
        "../notebooks/cellwhisperer_comprehensive_zero_shot.py.ipynb"

rule aggregate_comprehensive_benchmarks:
    """Aggregate all comprehensive benchmark results for spider plot integration."""
    input:
        benchmark_results=lambda wildcards: expand(
            rules.cellwhisperer_comprehensive_zero_shot.output.performance_metrics,
            zip,
            dataset=[pair[0] for pair in DATASET_METADATA_PAIRS],
            metadata_col=[pair[1] for pair in DATASET_METADATA_PAIRS],
            model=wildcards.model
        ) if DATASET_METADATA_PAIRS else []
    output:
        aggregated_comprehensive=COMPREHENSIVE_MODEL_RESULTS / "comprehensive_summary.csv"
    params:
        dataset_metadata_pairs=DATASET_METADATA_PAIRS,
        comprehensive_benchmarks=COMPREHENSIVE_BENCHMARKS
    conda:
        "cellwhisperer"
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=2"
    script:
        "../scripts/aggregate_comprehensive_benchmarks.py"

rule cellwhisperer_comprehensive_benchmark_all:
    """
    Run the comprehensive CellWhisperer benchmark end-to-end for all models.
    This provides the same evaluation as fig2 but integrated into spotwhisperer_eval.
    """
    input:
        # Summary results for trimodal and bimodal models
        expand(
            rules.aggregate_comprehensive_benchmarks.output.aggregated_comprehensive,
            model=[
                MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["trimodal"],
                MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["bimodal_matching"],
                MODEL_MAPPINGS["hest1k"]["bimodal_matching"], 
                MODEL_MAPPINGS["quilt1m"]["bimodal_matching"],
            ]
        ) if COMPREHENSIVE_BENCHMARKS else []