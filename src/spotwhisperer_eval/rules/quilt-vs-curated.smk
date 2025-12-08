# Quilt1M vs Quilt1M_curated comparison pipeline
# This file orchestrates comparison of bimodal bridge models trained with quilt1m vs quilt1m_curated
# to test whether quilt1m_curated improves "bridging" tasks

# NOTE: This file should be included from the main Snakefile which provides PROJECT_DIR and other common definitions

CURATED_COMPARISON_RESULTS = PROJECT_DIR / "results/quilt_vs_curated_comparison"

# Create modified model mappings with quilt1m_curated
MODEL_MAPPINGS_CURATED = {}
for test_dataset, mappings in MODEL_MAPPINGS.items():
    MODEL_MAPPINGS_CURATED[test_dataset] = {}
    for model_type, dataset_combo in mappings.items():
        # Replace quilt1m with quilt1m_curated in the dataset combination
        curated_combo = dataset_combo.replace("quilt1m", "quilt1m_curated")
        MODEL_MAPPINGS_CURATED[test_dataset][model_type] = curated_combo

rule quilt_curated_bimodal_bridge_plot:
    """
    Generate grouped bar plot comparing bimodal bridge models with quilt1m vs quilt1m_curated.
    Mimics the original bimodal_bridge_plot structure but shows grouped bars for comparison.
    """
    input:
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        # Original bimodal bridge model results (with quilt1m)
        original_retrieval_results=[
            rules.spotwhisperer_test.output.retrieval_and_cwevals.format(
                dataset_combo=MODEL_MAPPINGS[dataset]["bimodal_bridge"],
                test_dataset=dataset,
            )
            for dataset in BASE_DATASETS
        ],
        # Curated bimodal bridge model results (with quilt1m_curated)
        curated_retrieval_results=[
            rules.spotwhisperer_test.output.retrieval_and_cwevals.format(
                dataset_combo=MODEL_MAPPINGS_CURATED[dataset]["bimodal_bridge"],
                test_dataset=dataset.replace("quilt1m", "quilt1m_curated"),
            )
            for dataset in BASE_DATASETS
        ]
    output:
        plot=report(CURATED_COMPARISON_RESULTS / "bimodal_bridge_comparison.png", category="comparison", subcategory="quilt_curated", labels={"Analysis": "Bimodal bridge comparison", "Format": "plot"}),
        plot_svg=report(CURATED_COMPARISON_RESULTS / "bimodal_bridge_comparison.svg", category="comparison", subcategory="quilt_curated", labels={"Analysis": "Bimodal bridge comparison", "Format": "plot"}),
    params:
        datasets=BASE_DATASETS,
        model_mappings_original=MODEL_MAPPINGS,
        model_mappings_curated=MODEL_MAPPINGS_CURATED,
        modality_colors=MODALITY_COLORS,
        comparison_results_dir=CURATED_COMPARISON_RESULTS
    conda:
        "cellwhisperer"
    resources:
        mem_mb=20000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/quilt_curated_bimodal_bridge_comparison.ipynb"
    notebook:
        "../notebooks/quilt_curated_bimodal_bridge_comparison.py.ipynb"

rule quilt_curated_comparison_all:
    """
    Main rule to run complete quilt1m vs quilt1m_curated benchmark comparison.
    """
    input:
        rules.quilt_curated_bimodal_bridge_plot.output.plot
    default_target: True
