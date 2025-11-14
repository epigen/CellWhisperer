# HEST benchmark evaluation pipeline for SpotWhisperer
# Three-step pipeline per dataset: (1) patch extraction (2) inference (3) evaluation

# Define datasets to benchmark
HEST_DATASETS = ["IDC", "PRAD", "PAAD", "SKCM", "COAD", "READ", "CCRCC", "HCC", "LUNG", "LYMPH_IDC"]

# Path structure with dataset-specific folders using config paths
HEST_RESULTS = PROJECT_DIR / config["paths"]["hest_benchmark"]["results"]
HEST_MODEL_RESULTS = HEST_RESULTS / "{model}"
HEST_DATASET_RESULTS = HEST_MODEL_RESULTS / "{dataset}"

# Data paths from config
HEST_DATA_ROOT = PROJECT_DIR / config["paths"]["hest_benchmark"]["data_root"]
HEST_DATA_PREP = PROJECT_DIR / config["paths"]["hest_benchmark"]["data_processed"]
HEST_EMBED_ROOT = PROJECT_DIR / config["paths"]["hest_benchmark"]["embed_root"]

rule prepare_hest_data:
    """
    Prepare HEST benchmark data for a specific dataset using HEST conda environment

    This step downloads and prepares the HEST benchmark dataset
    with pre-extracted patches and split files.
    """
    input:
        config_file=PROJECT_DIR / "src/figures/config/bench_config.yaml"
    output:
        dataset_dir=directory(HEST_DATA_ROOT / "{dataset}")
    wildcard_constraints:
        dataset="(?!.*_spotwhisperer$).*"  # Don't match names ending with _spotwhisperer
    conda:
        "hest"
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=2",
        runtime=120  # 2 hours for downloading
    script:
        "../scripts/prepare_hest_data.py"

rule convert_hest_to_spotwhisperer:
    """
    Convert HEST benchmark dataset to standard SpotWhisperer dataset format

    This creates H5AD files with embedded images that can be used with
    the standard `cellwhisperer test` command instead of custom evaluation scripts.

    """
    input:
        dataset_dir=HEST_DATA_ROOT / "{dataset}"
    output:
        converted_dataset=PROJECT_DIR / config["paths"]["full_dataset"].replace("{dataset}", "hesteval_{dataset}"),
        # multi_folder=directory(PROJECT_DIR / config["paths"]["full_dataset_multi"].replace("{dataset}", "hesteval_{dataset}")),  # still contains _{i}
    conda:
        "cellwhisperer"
    params:
        multi_folder=lambda wildcards, output: Path(output.converted_dataset).parent / "h5ads",
    # Get H&E configuration for HEST datasets
        patch_size_pixels = config["he_configs"][config["dataset_he_mapping"]["hest1k"]]["patch_size_pixels"]
    resources:
        mem_mb=100000,
        slurm="cpus-per-task=4",
        runtime=180  # 3 hours for conversion
    script:
        "../scripts/convert_hest_to_spotwhisperer_dataset.py"

rule hest_spotwhisperer_test:
    """
    Run SpotWhisperer evaluation on converted HEST dataset using cellwhisperer test

    This uses the standard CellWhisperer testing pipeline on the converted dataset,
    eliminating the need for custom inference and evaluation scripts.
    """
    input:
        converted_dataset=rules.convert_hest_to_spotwhisperer.output.converted_dataset,
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
        base_config=ancient(PROJECT_DIR / "src/spotwhisperer_v3.yaml")
    output:
        # Follow same pattern as spotwhisperer_test rule
        results_csv=PROJECT_DIR / config["paths"]["csv_logs"] / "hest_eval___{model}___{dataset}" / "metrics.csv"
    params:
        test_dataset="hesteval_{dataset}",
        batch_size=64,
        seed=1,
        project_dir=PROJECT_DIR
    conda:
        "cellwhisperer"
    resources:
        slurm=slurm_gres("small", num_cpus=2),
        mem_mb=100000,
        runtime=120  # 2 hours for evaluation
    shell:
        """
        cd {params.project_dir}
        cellwhisperer test \
            --config {input.base_config} \
            --model_ckpt {input.model} \
            --data.dataset_names {params.test_dataset} \
            --data.train_fraction 0.0 \
            --batch_size {params.batch_size} \
            --seed_everything {params.seed} \
            --nproc 0 \
            --wandb '' \
            --trainer.logger.init_args.version hest_eval___{wildcards.model}___{wildcards.dataset}
        """


rule aggregate_hest_results:
    """
    Aggregate results from all datasets into a single summary
    """
    input:
        results_csvs=expand(rules.hest_spotwhisperer_test.output.results_csv,
               dataset=HEST_DATASETS,
               allow_missing=True
        )
    output:
        aggregated_summary=HEST_MODEL_RESULTS / "aggregated_benchmark_summary.json"
    params:
        datasets=HEST_DATASETS,
        metrics=[
            "test_retrieval/transcriptome_image/rocauc_macroAvg",  # aggregate over this one for "overall_performance"
            # "test_retrieval/image_transcriptome/rocauc_macroAvg",  # TODO enable later (needs recompute)
            "test_retrieval/transcriptome_image/f1_macroAvg",
            # "test_retrieval/image_transcriptome/f1_macroAvg",  # TODO enable later (needs recompute)
        ]
    conda:
        "cellwhisperer"
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/aggregate_hest_results.py"

rule hest_per_class_analysis:
    """
    Generate per-class analysis comparing trimodal vs bimodal models
    for HEST benchmark datasets (10 organs/9 cancers)
    """
    input:
        # Results from trimodal and bimodal_matching models for HEST datasets
        lambda wildcards: [
            PROJECT_DIR / config["paths"]["csv_logs"] / "hest_eval___spotwhisperer_{}___{}".format(combo, dataset) / "metrics.csv"
            for combo in ["cellxgene_census__archs4_geo__hest1k__quilt1m",  # trimodal
                         "cellxgene_census__archs4_geo", "hest1k", "quilt1m"]  # bimodal matching options
            for dataset in HEST_DATASETS
        ]
    output:
        analysis=report(HEST_RESULTS / "comparison" / "per_class_analysis.csv", category="per_class_analysis", subcategory="transcriptome-image", labels={"Analysis": "HEST Benchmark (retrieval-based)", "Format": "csv"}),
        plot=report(HEST_RESULTS / "comparison" / "per_class_analysis.pdf", category="per_class_analysis", subcategory="transcriptome-image", labels={"Analysis": "HEST Benchmark (retrieval-based)", "Format": "plot"}),
        clip_scores=report(HEST_RESULTS / "comparison" / "test_individual_clip_scores.csv", category="per_class_analysis", subcategory="transcriptome-image", labels={"Analysis": "HEST Benchmark (retrival-based)", "Format": "csv (CLIP scores)"})
    params:
        datasets=HEST_DATASETS,
        model_types=["trimodal", "bimodal_mismatch1", "bimodal_mismatch2", "bimodal_matching"]
    conda:
        "cellwhisperer"
    resources:
        mem_mb=20000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/hest_per_class_analysis.ipynb"
    notebook:
        "../notebooks/hest_per_class_analysis.py.ipynb"

# Main rule to run complete HEST benchmark across all datasets
rule hest_benchmark_all:
    input:
        expand(
            rules.aggregate_hest_results.output.aggregated_summary,
            model=config["model_name_path_map"]["spotwhisperer3"]
        ),
        # Per-class analysis
        rules.hest_per_class_analysis.output.analysis
    default_target: True
