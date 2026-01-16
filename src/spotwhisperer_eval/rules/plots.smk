# Plotting rules for aggregated analyses

rule create_analysis_plots:
    """
    Generate comparison barplots across metrics for four model types per test dataset.
    """
    input:
        hest_results=lambda wildcards: expand(
            rules.aggregate_hest_evaluation.output.aggregated_hest,
            dataset_combo=[MODEL_MAPPINGS[wildcards.test_dataset][mt] for mt in ["naive_baseline", "bimodal_matching", "bimodal_bridge", "trimodal"]],
            test_dataset=wildcards.test_dataset,
        ),
        musk_results=lambda wildcards: expand(
            rules.aggregate_musk_results.output.aggregated_musk,
            dataset_combo=[MODEL_MAPPINGS[wildcards.test_dataset][mt] for mt in ["naive_baseline", "bimodal_matching", "bimodal_bridge", "trimodal"]],
            test_dataset=wildcards.test_dataset,
        ),
        retrieval_results=lambda wildcards: expand(
            rules.aggregate_spotwhisperer_test.output.aggregated_retrieval,
            dataset_combo=[MODEL_MAPPINGS[wildcards.test_dataset][mt] for mt in ["naive_baseline", "bimodal_matching", "bimodal_bridge", "trimodal"]],
            test_dataset=wildcards.test_dataset,
        ),
        cwevals_results=lambda wildcards: expand(
            rules.aggregate_spotwhisperer_test.output.aggregated_cwevals,
            dataset_combo=[MODEL_MAPPINGS[wildcards.test_dataset][mt] for mt in ["naive_baseline", "bimodal_matching", "bimodal_bridge", "trimodal"]],
            test_dataset=wildcards.test_dataset,
        )
    output:
        plots=report(BENCHMARKS_DIR / "analysis_plots" / "{test_dataset}" / "performance_comparison.png", category="aggregated", subcategory=lambda wildcards: DATASET_PAIR_MAPPING[wildcards.test_dataset], labels={"Analysis": "Overview", "Format": "plot"}),
    params:
        model_mappings=lambda wildcards: MODEL_MAPPINGS[wildcards.test_dataset],
        benchmarks_dir=BENCHMARKS_DIR,
        dataset_combos=DATASET_COMBOS,
        metrics=[
            "hest",
            "pannuke",
            "skin",
            "valfn_zshot_TabSap_cell_lvl/f1_macroAvg",
            "valfn_zshot_TabSap_cell_lvl/rocauc_macroAvg",
        ],
    conda:
        "cellwhisperer"
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=4"
    notebook:
        "../notebooks/create_analysis_plots.py.ipynb"

rule create_retrieval_plots:
    """
    Plot retrieval metrics comparing naive baseline and bimodal bridge models.
    """
    input:
        retrieval_results=lambda wildcards: expand(
            rules.aggregate_spotwhisperer_test.output.aggregated_retrieval,
            dataset_combo=[MODEL_MAPPINGS[wildcards.test_dataset][mt] for mt in ["naive_baseline", "bimodal_bridge"]],
            test_dataset=wildcards.test_dataset,
        )
    output:
        plots=report(BENCHMARKS_DIR / "retrieval_plots" / "{test_dataset}" / "retrieval_comparison.png", category="aggregated", subcategory=lambda wildcards: DATASET_PAIR_MAPPING[wildcards.test_dataset], labels={"Analysis": "Retrieval"}),
    params:
        model_mappings=lambda wildcards: {k: v for k, v in MODEL_MAPPINGS[wildcards.test_dataset].items() if k in ["naive_baseline", "bimodal_bridge"]},
        benchmarks_dir=BENCHMARKS_DIR,
        retrieval_metrics=[
            "test/clip_loss_epoch",
            "test_retrieval/left_right/recall_at_50_macroAvg",
            "test_retrieval/right_left/recall_at_50_macroAvg",
            "test_retrieval/left_right/rocauc_macroAvg",
            "test_retrieval/right_left/rocauc_macroAvg",
        ],
    conda:
        "cellwhisperer"
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=4"
    notebook:
        "../notebooks/create_retrieval_plots.py.ipynb"

rule spider_performance_plot:
    """
    Radar plot comparing trimodal and modality-specific bimodal models across grouped metrics.
    """
    input:
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        hest_results=expand(
            rules.aggregate_hest_evaluation.output.aggregated_hest,
            dataset_combo=[
                MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["trimodal"],
                MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["bimodal_matching"],
                MODEL_MAPPINGS["hest1k"]["bimodal_matching"],
                MODEL_MAPPINGS["quilt1m"]["bimodal_matching"],
            ],
        ),
        musk_results=expand(
            rules.aggregate_musk_results.output.aggregated_musk,
            dataset_combo=[
                MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["trimodal"],
                MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["bimodal_matching"],
                MODEL_MAPPINGS["hest1k"]["bimodal_matching"],
                MODEL_MAPPINGS["quilt1m"]["bimodal_matching"],
            ],
        ),
        retrieval_results=expand(
            rules.aggregate_spotwhisperer_test.output.aggregated_retrieval,
            dataset_combo=[
                MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["trimodal"],
                MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["bimodal_matching"],
                MODEL_MAPPINGS["hest1k"]["bimodal_matching"],
                MODEL_MAPPINGS["quilt1m"]["bimodal_matching"],
            ],
        ),
        cwevals_results=expand(
            rules.aggregate_spotwhisperer_test.output.aggregated_cwevals,
            dataset_combo=[
                MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["trimodal"],
                MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["bimodal_matching"],
                MODEL_MAPPINGS["hest1k"]["bimodal_matching"],
                MODEL_MAPPINGS["quilt1m"]["bimodal_matching"],
            ],
        ),
        # comprehensive_results: Using existing validation metrics instead
        # expand(
        #     rules.aggregate_comprehensive_benchmarks.output.aggregated_comprehensive,
        #     model=[
        #         "spotwhisperer_{}".format(MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["trimodal"]),
        #         "spotwhisperer_{}".format(MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["bimodal_matching"]),
        #         "spotwhisperer_{}".format(MODEL_MAPPINGS["hest1k"]["bimodal_matching"]),
        #         "spotwhisperer_{}".format(MODEL_MAPPINGS["quilt1m"]["bimodal_matching"]),
        #     ],
        # ),
        # lung_results=expand(
        #     BENCHMARKS_DIR / "lung" / "{dataset_combo}" / "performance_summary.csv",
        #     dataset_combo=[
        #         MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["trimodal"],
        #         MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["bimodal_matching"],
        #         MODEL_MAPPINGS["hest1k"]["bimodal_matching"],
        #         MODEL_MAPPINGS["quilt1m"]["bimodal_matching"],
        #     ],
        # ),
        # TODO not yet functional 
        # pathocell_results=expand(
        #     rules.aggregate_pathocell_results.output.aggregated_pathocell,
        #     dataset_combo=[
        #         MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["trimodal"],
        #         MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["bimodal_matching"],
        #         MODEL_MAPPINGS["hest1k"]["bimodal_matching"],
        #         MODEL_MAPPINGS["quilt1m"]["bimodal_matching"],
        #     ],
        # )
    output:
        plot=report(BENCHMARKS_DIR / "spider_plot" / "model_comparison_radar.png", category="aggregated", subcategory="all_modalities", labels={"Analysis": "Radar comparison", "Format": "plot"}),
    params:
        model_configs=[
            ("trimodal", MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["trimodal"]),
            ("text-transcriptome", MODEL_MAPPINGS["cellxgene_census__archs4_geo"]["bimodal_matching"]),
            ("image-transcriptome", MODEL_MAPPINGS["hest1k"]["bimodal_matching"]),
            ("image-text", MODEL_MAPPINGS["quilt1m"]["bimodal_matching"]),
        ],
        metrics_by_modality={
            "text-image": ["pannuke", "skin", "lung_tissue_region_type_expert_annotation_accuracy", "lung_tissue_cell_type_annotations_accuracy", "pathocell_image_text_retrieval", "pathocell_zero_shot_classification"],
            "image-transcriptome": [f"hest_{dataset}" for dataset in HEST_DATASETS] + ["pathocell_embedding_quality"],
            "text-transcriptome": [
                "valfn_zshot_TabSap_cell_lvl/f1_macroAvg",
                "valfn_zshot_TabSap_cell_lvl/rocauc_macroAvg",
                # Use existing comprehensive validation metrics that are already available
                "valfn_zshot_HumanDisease_disease_subtype/f1_macroAvg"
                "valfn_zshot_HumanDisease_disease_subtype/rocauc_macroAvg"
                # "valfn_human_disease_strictly_deduplicated_dmis-lab_biobert-v1.1_CLS_pooling/text_as_classes_f1_macroAvg",
                # "valfn_human_disease_strictly_deduplicated_dmis-lab_biobert-v1.1_CLS_pooling/text_as_classes_rocauc_macroAvg",
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
    notebook:
        "../notebooks/spider_performance_plot.py.ipynb"

rule bimodal_bridge_plot:
    """
    AUROC comparison for bimodal bridge models on their respective datasets with baseline.
    """
    input:
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        retrieval_results=[
            rules.spotwhisperer_test.output.retrieval_and_cwevals.format(
                dataset_combo=MODEL_MAPPINGS[dataset]["bimodal_bridge"],
                test_dataset=dataset,
            )
            for dataset in BASE_DATASETS
        ],
    output:
        plot=report(BENCHMARKS_DIR / "bimodal_bridge_plot" / "auroc_comparison.png", category="aggregated", subcategory="bimodal_bridge", labels={"Analysis": "Bimodal bridge AUROC", "Format": "plot"}),
        plot_svg=report(BENCHMARKS_DIR / "bimodal_bridge_plot" / "auroc_comparison.svg", category="aggregated", subcategory="bimodal_bridge", labels={"Analysis": "Bimodal bridge AUROC", "Format": "plot"}),
    params:
        datasets=BASE_DATASETS,
        model_mappings=MODEL_MAPPINGS,
        modality_colors=MODALITY_COLORS,
    conda:
        "cellwhisperer"
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=4"
    notebook:
        "../notebooks/bimodal_bridge_plot.py.ipynb"
