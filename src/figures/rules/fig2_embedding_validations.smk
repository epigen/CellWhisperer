include: "../../shared/rules/training_sample_weights.smk"

ZERO_SHOT_RESULTS = PROJECT_DIR / config["paths"]["zero_shot_validation"]["result_dir"]


# Extended fig 2
rule cw_transcriptome_term_scores:
    """
    - Compute the term-based match scores using (CW)
        - Embed all terms (present in gsva_results)
        - Use processed dataset and get cos_sim between each cell and each term
    """
    input:
        processed_dataset=rules.process_full_dataset.output.model_outputs,  # PROJECT_DIR / config["paths"]["model_processed_dataset"],
        gsva_results=PROJECT_DIR / config["paths"]["gsva"]["result"],
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",  # needed for the keywords
    output:
        cw_transcriptome_term_scores=PROJECT_DIR / config["paths"]["gsva"]["cw_transcriptome_term_scores"],
    conda:
        "cellwhisperer"
    resources:
        mem_mb=40000,
        slurm="cpus-per-task=5 gres=gpu:a100:1 qos=a100 partition=gpu"
    log:
        notebook="../logs/gsva_correlation_{model}_{dataset}.log"
    notebook:
        "../notebooks/cw_transcriptome_term_scores.py.ipynb"


rule plot_gsva_correlations:
    """
    """
    input:
        cw_transcriptome_term_scores=PROJECT_DIR / config["paths"]["gsva"]["cw_transcriptome_term_scores"],
        gsva_results=PROJECT_DIR / config["paths"]["gsva"]["result"],
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        gsva_correlation_results=PROJECT_DIR / config["paths"]["gsva"]["correlation"],
        top_term_correlation=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_top_term_correlation.svg",
        library_correlations=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_library_correlations.svg",
        term_level_correlations=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_term_level_correlations.svg",
        omim_correlations=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_omim_correlations.svg",
        cw_binarized_gsva_scores=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "cw_binarized_gsva_scores.svg",
        library_ks_statistics=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_library_ks_statistics.svg",  # also binarized
        cherry_picked_examples=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_cherry_picked_examples.svg",
    params:
        selected_top_term="colorectal cancer"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=200000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/plot_gsva_correlation_{dataset}_{model}.ipynb"
    notebook:
        "../notebooks/plot_gsva_correlations.py.ipynb"

# Fig 2
rule zero_shot_validations:
    """
    For a given dataset, produce: 
     - performance evaluation metrics (macro-averaged and per class) for various label predictions
     - cell type reference and predictions on UMAP
     - confusion matrix
     - integration scores
    """
    input:
        # no need to have seperate embeddings and read count tables for the tabula sapiens well studied cell types, can just use the full dataset and subset:
        processed_dataset=lambda wildcards: rules.process_full_dataset.output.model_outputs.format(dataset=wildcards.dataset if wildcards.dataset != "tabula_sapiens_well_studied_celltypes" else "tabula_sapiens", model=wildcards.model),
        raw_read_count_table=lambda wildcards: str(PROJECT_DIR / config["paths"]["read_count_table"]).format(dataset=wildcards.dataset if wildcards.dataset != "tabula_sapiens_well_studied_celltypes" else "tabula_sapiens"),
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",  # needed to embed the keywords  
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        # Using a directory here because the exact files produced depend on the dataset:
        output_directory=directory(ZERO_SHOT_RESULTS / "datasets" / "{dataset,[^/]+}"),
    params:
        dataset = config["zero_shot_validation_datasets"],
        metadata_cols_per_dataset = config["metadata_cols_per_zero_shot_validation_dataset"],
        transcriptome_model_name = "geneformer"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=350000,
        slurm=f"cpus-per-task=5 gres=gpu:{GPU_TYPE}:1 qos={GPU_TYPE} partition=gpu"

    log:
        notebook="../logs/zero_shot_validation_{model}_{dataset}.ipynb"
    notebook:
        "../notebooks/zero_shot_validation.py.ipynb"


rule performance_macroavg_and_example_plots:
    """
    Summarize the performance metrics across multiple datasets and metadata columns
    """
    input:
        zero_shot_validations_result_dirs=[ZERO_SHOT_RESULTS / "datasets" / dataset for dataset in config["zero_shot_validation_datasets"]],
    output:
        macrovag_summary_plot=ZERO_SHOT_RESULTS / "performance_metrics_cellwhisperer.selected_datasets.rocauc_and_accuracy.pdf",
        per_class_examples_plot=ZERO_SHOT_RESULTS / "performance_metrics_cellwhisperer.selected_classes_and_datasets.pdf",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=2000,
        slurm="cpus-per-task=1"
    log:
        notebook="../logs/performance_macroavg_and_example_plots_{model}.ipynb"
    notebook:
        "../notebooks/performance_macroavg_and_example_plots.py.ipynb"


