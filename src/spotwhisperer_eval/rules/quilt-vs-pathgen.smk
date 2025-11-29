# Quilt1M vs PathGen comparison pipeline
# This file orchestrates training and evaluation of bimodal models for each dataset,
# then generates paired comparison plots across all benchmarks

# Define the two dataset combinations to compare
COMPARISON_DATASETS = ["pathgen", "quilt1m"]
COMPARISON_RESULTS = PROJECT_DIR / "results/quilt_vs_pathgen_comparison"


rule quilt_pathgen_paired_barplots:
    """
    Generate paired barplots comparing quilt1m vs pathgen across all benchmarks.

    NOTE: Untested!!
    """
    input:
        # Actual result files for plotting
        lung_results=expand(
            SPOTWHISPERER_EVAL_RESULTS / "lung/performance_summary_spotwhisperer_{dataset}.csv",
            dataset=COMPARISON_DATASETS
        ),
        # pathocell_results=expand(
        #     BENCHMARKS_DIR / "pathocell" / "{dataset}" / "performance_summary.json",
        #     dataset=COMPARISON_DATASETS
        # ),
        musk_results=expand(
            BENCHMARKS_DIR / "musk" / "{dataset}" / "performance_summary.json", 
            dataset=COMPARISON_DATASETS
        ),
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        plots=COMPARISON_RESULTS / "paired_barplots.png",
        plots_svg=COMPARISON_RESULTS / "paired_barplots.svg",
        summary_table=COMPARISON_RESULTS / "comparison_summary.csv"
    params:
        datasets=COMPARISON_DATASETS,
        benchmarks=["lung", "musk"],  # "pathocell", 
        comparison_results_dir=COMPARISON_RESULTS
    conda: 
        "cellwhisperer"
    resources:
        mem_mb=20000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/quilt_pathgen_comparison_plots.ipynb"
    notebook:
        "../notebooks/quilt_pathgen_comparison_plots.py.ipynb"

rule quilt_pathgen_comparison_all:
    """
    Main rule to run complete quilt1m vs pathgen benchmark comparison.
    """
    input:
        rules.quilt_pathgen_paired_barplots.output.plots,
        rules.quilt_pathgen_paired_barplots.output.summary_table
    default_target: True
