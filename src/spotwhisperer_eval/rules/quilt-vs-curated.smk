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


rule quilt_curated_trimodal_spider_plot:
    """
    Radar plot comparing trimodal vs trimodal_curated across grouped metrics.
    Uses same metric groups as spider_performance_plot but overlays two trimodal models.
    """
    input:
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        hest_results=expand(
            rules.aggregate_hest_evaluation.output.aggregated_hest,
            dataset_combo=[
                MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["trimodal"],
                MODEL_MAPPINGS_CURATED["cellxgene_census__archs4_geo"]["trimodal"],
            ],
        ),
        musk_results=expand(
            rules.aggregate_musk_results.output.aggregated_musk,
            dataset_combo=[
                MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["trimodal"],
                MODEL_MAPPINGS_CURATED["cellxgene_census__archs4_geo"]["trimodal"],
            ],
        ),
        cwevals_results=expand(
            rules.aggregate_spotwhisperer_test.output.aggregated_cwevals,
            dataset_combo=[
                MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["trimodal"],
                MODEL_MAPPINGS_CURATED["cellxgene_census__archs4_geo"]["trimodal"],
            ],
        )
    output:
        plot=report(CURATED_COMPARISON_RESULTS / "trimodal_spider_comparison.png", category="comparison", subcategory="quilt_curated", labels={"Analysis": "Trimodal spider comparison", "Format": "plot"}),
        plot_svg=report(CURATED_COMPARISON_RESULTS / "trimodal_spider_comparison.svg", category="comparison", subcategory="quilt_curated", labels={"Analysis": "Trimodal spider comparison", "Format": "plot"}),
    params:
        model_configs=[
            ("trimodal", MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["trimodal"]),
            ("trimodal_curated", MODEL_MAPPINGS_CURATED["cellxgene_census__archs4_geo"]["trimodal"]),
        ],
        metrics_by_modality={
            "text-image": [
                "pannuke",
                "skin",
                "lung_tissue_region_type_expert_annotation_accuracy",
                "lung_tissue_cell_type_annotations_accuracy",
                "pathocell_image_text_retrieval",
                "pathocell_zero_shot_classification",
            ],
            "image-transcriptome": [f"hest_{dataset}" for dataset in HEST_DATASETS] + ["pathocell_embedding_quality"],
            "text-transcriptome": [
                "valfn_zshot_TabSap_cell_lvl/f1_macroAvg",
                "valfn_zshot_TabSap_cell_lvl/rocauc_macroAvg",
                "valfn_human_disease_strictly_deduplicated_dmis-lab_biobert-v1.1_CLS_pooling/text_as_classes_f1_macroAvg",
                "valfn_human_disease_strictly_deduplicated_dmis-lab_biobert-v1.1_CLS_pooling/text_as_classes_rocauc_macroAvg",
                "valfn_immgen_deduplicated/text_as_classes_f1_macroAvg",
                "valfn_immgen_deduplicated/text_as_classes_rocauc_macroAvg",
                "valfn_zshot_TabSapWellStudied_cell_lvl/f1_macroAvg",
            ],
        },
        modality_colors=MODALITY_COLORS,
    conda:
        "cellwhisperer"
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=4"
    log:
        notebook="../logs/quilt_curated_trimodal_spider_plot.ipynb"
    notebook:
        "../notebooks/quilt_curated_trimodal_spider_plot.py.ipynb"


rule quilt_curated_individual_score_violins:
    """
    Violin plots of individual CLIP scores for quilt1m vs quilt1m_curated.
    Loads two CSVs written by spotwhisperer_test and compares distributions.
    """
    input:
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        individual_scores=[
            PROJECT_DIR / config["paths"]["csv_logs"] / "sweval___{}___{}".format(MODEL_MAPPINGS["quilt1m"]["bimodal_bridge"], td) / "test_individual_clip_scores.csv"
            for td in ["quilt1m", "quilt1m_curated"]
        ]
    output:
        plot=report(CURATED_COMPARISON_RESULTS / "violin_distributions" / "individual_clip_score_violins.svg", category="comparison", subcategory="quilt_curated", labels={"Analysis": "CLIP score distributions", "Format": "plot"}),
        stats=CURATED_COMPARISON_RESULTS / "violin_distributions" / "individual_clip_score_violins_stats.txt"
    params:
        dataset_combo=MODEL_MAPPINGS["quilt1m"]["bimodal_bridge"],
        test_datasets=["quilt1m", "quilt1m_curated"],
    conda:
        "cellwhisperer"
    resources:
        mem_mb=15000,
        slurm="cpus-per-task=1"
    script:
        "../notebooks/quilt_curated_individual_score_violins.py"
