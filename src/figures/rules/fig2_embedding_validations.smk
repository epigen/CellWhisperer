

ZERO_SHOT_RESULTS = PROJECT_DIR / "results/plots/zero_shot_validation"
ZERO_SHOT_MODEL_RESULTS = ZERO_SHOT_RESULTS / "{model,cellwhisperer_clip_v1}"  # TODO add other models here

ZERO_SHOT_PREDICTORS = [
        "gpt4", 
        "llama33", 
        "cellwhisperer_clip_v1",
        # "claudesonnet"  # TODO consider also opus, as it was best in another benchmark..
        ] 

include: "../../shared/rules/training_sample_weights.smk"

# Computations
rule compute_umap_neighbors:
    """
    # TODO generalize towards others models (beyond cellwhisperer_clip_v1)
    # TODO might need to move to another include
    """
    input:
        processed_dataset=rules.process_full_dataset.output.model_outputs,
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"]
    output:
        umap=ZERO_SHOT_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "X_umap_on_neighbors_{model}.npz"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=200000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/compute_umap_neighbors_{dataset}_{model}.ipynb"
    notebook:
        "../notebooks/compute_umap_neighbors.py.ipynb"

include: "zero_shot_llm.smk"
include: "zero_shot_finetuning.smk"

rule zero_shot_cellwhisperer_prediction:
    """
    Zero-shot property prediction with CellWhisperer

    For a given dataset and field, compute all predictions for downstream analysis

    NOTE: could be used as template for few-shot learing with other embedding-based model zero shot predictions
    """
    input:
        # no need to have seperate embeddings and read count tables for the tabula sapiens well studied cell types, can just use the full dataset and subset:
        processed_dataset=rules.process_full_dataset.output.model_outputs,
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",  # needed to embed the keywords  
    output:
        # Using a directory here because the exact files produced depend on the dataset:
        predictions=ZERO_SHOT_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "predictions" / "{metadata_col}.{grouping,by_cell|by_class}.csv",  # NOTE: might be used as input for `plot_confusion_matrix` as well (and by extension `zero_shot_performance_macroavg`)
        # scores=ZERO_SHOT_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "predictions" / "{metadata_col}.{grouping}.scores.csv",  # TODO refactor away since everything is in predictions anyways
    params:
        use_prefix_suffix_version=True,
        average_by_class=lambda wildcards: wildcards.grouping == "by_class",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=350000,
        slurm=f"cpus-per-task=5 gres=gpu:{GPU_TYPE}:1 qos={GPU_TYPE} partition=gpu"
    log:
        notebook="../logs/zero_shot_cellwhisperer_prediction_{model}_{dataset}_{metadata_col}_{grouping}.ipynb"
    notebook:
        "../notebooks/zero_shot_cellwhisperer_prediction.py.ipynb"


# Embedding analysis using scIB (Extended data figure)
rule transcriptome_embedding_scib:
    """
    Embedding integration scores, computed using the `scib` package
    """
    input:
        # no need to have seperate embeddings and read count tables for the tabula sapiens well studied cell types, can just use the full dataset and subset:
        processed_dataset=rules.process_full_dataset.output.model_outputs,
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        umap=rules.compute_umap_neighbors.output.umap,
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
    output:
    # /mnt/muwhpc/cellwhisperer_private/results/plots/zero_shot_validation/cellwhisperer_clip_v1/datasets/tabula_sapiens_well_studied_celltypes
        embedding_plots_zero_shot_comparison_pdf=ZERO_SHOT_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "embedding_plots_zero_shot_comparison.pdf",
        embedding_plots_zero_shot_comparison_png=ZERO_SHOT_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "embedding_plots_zero_shot_comparison.png",
        integration_scores=ZERO_SHOT_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "embedding_scib_scores.json",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=350000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/transcriptome_embeddings_scib_{model}_{dataset}.ipynb"
    notebook:
        "../notebooks/transcriptome_embedding_scib.py.ipynb"

rule plot_joint_transcriptome_embedding_scib:
    """Bar plots of integration metrics for all methods."""
    input:
        # TODO all the different models to cross-check
        [rules.transcriptome_embedding_scib.output.integration_scores.replace("{model}", model) for model in ["cellwhisperer_clip_v1"]]  # TODO "geneformer", 
    output:
        integration_scores=ZERO_SHOT_RESULTS / "integration_scores_{dataset}.csv",
        integration_scores_plot=ZERO_SHOT_RESULTS / "integration_scores_{dataset}.pdf"
    conda:
        "cellwhisperer"
    params:
        models=["geneformer", "cellwhisperer"]
    resources:
        mem_mb=2000,
        slurm="cpus-per-task=1"
    log:
        notebook="../logs/plot_joint_transcriptome_embedding_scib_{dataset}.ipynb"
    notebook:
        "../notebooks/plot_joint_transcriptome_embedding_scib.py.ipynb"


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


rule plot_confidence_distributions:
    """
    Plot a number of histograms and KDEplots for the cellwhisperer score across different values for metadata_col

    """
    input:
        processed_dataset=rules.process_full_dataset.output.model_outputs,
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        predictions=rules.zero_shot_cellwhisperer_prediction.output.predictions.replace("{grouping}", "by_cell"),
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        plot_dir=directory(ZERO_SHOT_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "confidence_distribution_plots")  # NOTE: could be disentangled as well
    conda:
        "cellwhisperer"
    params:
        transcriptome_model_name = "geneformer"
    resources:
        mem_mb=500000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/plot_confidence_distributions_{model}_{dataset}_{metadata_col}.ipynb"
    notebook:
        "../notebooks/plot_confidence_distributions.py.ipynb"

rule plot_zero_shot_predictions_on_umap:
    """
    Generate a series of plots that summarize the zero-shot validation results
    """
    input:
        processed_dataset=rules.process_full_dataset.output.model_outputs,
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        predictions=rules.zero_shot_cellwhisperer_prediction.output.predictions.replace("{grouping}", "by_cell"),
        umap=rules.compute_umap_neighbors.output.umap,
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        result_dir=directory(ZERO_SHOT_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "other_plots"),
    conda:
        "cellwhisperer"
    params:
        transcriptome_model_name = "geneformer",
        use_prefix_suffix_version=True
    resources:
        mem_mb=400000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/plot_zero_shot_predictions_on_umap_{model}_{dataset}_{metadata_col}.ipynb"
    notebook:
        "../notebooks/plot_zero_shot_predictions_on_umap.py.ipynb"

rule plot_confusion_matrix:
    """
    NOTE: could probably use `scores` computed by `zero_shot_cellwhisperer_prediction`. This would work by refactoring the bloated function `get_performance_metrics_transcriptome_vs_text`, such that it takes as input only the scores and then computes the metrics.
    """
    input:
        predictions=rules.zero_shot_cellwhisperer_prediction.output.predictions.replace("{grouping}", "by_cell"),
        processed_dataset=rules.process_full_dataset.output.model_outputs,
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt"
    output:
        confusion_matrix_plot=ZERO_SHOT_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "confusion_matrix_{normed}.pdf",
        confusion_matrix_table=ZERO_SHOT_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "confusion_matrix_{normed}.xlsx",
        performance_metrics=ZERO_SHOT_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "performance_metrics_{normed}.csv",  # note that this is the same for normed and unnormed (would be better to split the rule once more actually)
        performance_metrics_per_metadata=ZERO_SHOT_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "performance_metrics_per_metadata_{normed}.csv",
    params:
        transcriptome_model_name = "geneformer",
        normed=lambda wildcards: wildcards.normed == "normed",
        use_prefix_suffix_version=True
    resources:
        mem_mb=350000,
        slurm=f"cpus-per-task=5 gres=gpu:{GPU_TYPE}:1 qos={GPU_TYPE} partition=gpu"
    log:
        notebook="../logs/plot_confusion_matrix_{model}_{dataset}_{metadata_col}_{normed}.ipynb"
    notebook:
        "../notebooks/plot_confusion_matrix.py.ipynb"

rule plot_term_search_results:
    """
    Plot the ground truth celltype and the keyword search results on the UMAP (Fig 2a)
    """
    input:
        # no need to have seperate embeddings and read count tables for the tabula sapiens well studied cell types, can just use the full dataset and subset:
        processed_dataset=lambda wildcards: rules.process_full_dataset.output.model_outputs.format(dataset="tabula_sapiens", model=wildcards.model),
        raw_read_count_table=str(PROJECT_DIR / config["paths"]["read_count_table"]).format(dataset="tabula_sapiens"),
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",  # needed to embed the keywords
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        umap=rules.compute_umap_neighbors.output.umap.replace("{dataset}", "tabula_sapiens")
    output:
        # Using a directory here because the exact files produced depend on the dataset:
        umap_on_neighbors_celltype=ZERO_SHOT_MODEL_RESULTS / "datasets" / "tabula_sapiens" / "umap_on_neighbors_cellwhisperer.true_celltype_{celltype}.{file_suffix}",
        colorscale_symmetrical=ZERO_SHOT_MODEL_RESULTS / "datasets" / "tabula_sapiens" / "umap_on_neighbors_cellwhisperer.keyword_for_{celltype}.symmetrical_cmap.{file_suffix}",
        colorscale_asymmetrical=ZERO_SHOT_MODEL_RESULTS / "datasets" / "tabula_sapiens" / "umap_on_neighbors_cellwhisperer.keyword_for_{celltype}.asymmetrical_cmap.{file_suffix}",
    params:
        celltype_terms_dict=CELLTYPE_TERMS_DICT,
        suffix_prefix_dict=SUFFIX_PREFIX_DICT,
        dataset = "tabula_sapiens",
        transcriptome_model_name = "geneformer"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=350000,
        slurm=f"cpus-per-task=5 gres=gpu:{GPU_TYPE}:1 qos={GPU_TYPE} partition=gpu"
    log:
        notebook="../logs/plot_term_search_results_{model}_{celltype}_tabula_sapiens_{file_suffix}.ipynb"
    notebook:
        "../notebooks/plot_term_search_results.py.ipynb"

rule zero_shot_performance_macroavg:
    """
    Aggregated performance metrics (barplots) across multiple datasets and metadata columns

    Ugly but results are correct
    """
    input:
        datasets_macroavg=expand(rules.plot_confusion_matrix.output.performance_metrics, zip,
                                 dataset=["tabula_sapiens", "tabula_sapiens_well_studied_celltypes", "pancreas", "immgen", "human_disease", "tabula_sapiens", "human_disease"],
                                 metadata_col=["celltype", "celltype", "celltype", "celltype", "Disease_subtype", "organ_tissue", "Tissue"],
                                 model=["{model}", "{model}", "{model}", "{model}", "{model}", "{model}", "{model}"],
                                 normed=["raw", "raw", "raw", "raw", "raw", "raw", "raw"] # normed/raw are the same :|
                                 ),
        datasets_perlabel=expand(rules.plot_confusion_matrix.output.performance_metrics_per_metadata, zip,
                                 dataset=["tabula_sapiens", "tabula_sapiens_well_studied_celltypes", "pancreas", "immgen", "human_disease", "tabula_sapiens", "human_disease"],
                                 metadata_col=["celltype", "celltype", "celltype", "celltype", "Disease_subtype", "organ_tissue", "Tissue"],
                                 model=["{model}", "{model}", "{model}", "{model}", "{model}", "{model}", "{model}"],
                                 normed=["raw", "raw", "raw", "raw", "raw", "raw", "raw"]  # normed/raw are the same :|
                                 ),
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        macroavg_summary_plot=ZERO_SHOT_MODEL_RESULTS / "performance_metrics.selected_datasets.rocauc_and_accuracy.pdf",
    conda:
        "cellwhisperer"
    params:
        label_cols=[
            "celltype",
            "celltype",
            "celltype",
            "celltype",
            "Disease_subtype",
            "organ_tissue",
            "Tissue",],
        datasets=["tabula_sapiens", "tabula_sapiens_well_studied_celltypes", "pancreas", "immgen", "human_disease", "tabula_sapiens", "human_disease"]
    resources:
        mem_mb=2000,
        slurm="cpus-per-task=1"
    log:
        notebook="../logs/zero_shot_performance_macroavg_{model}.ipynb"
    notebook:
        "../notebooks/zero_shot_performance_macroavg.py.ipynb"

rule zero_shot_performance_examples:
    """
    Example accuracies for zero-shot cell type property predictions
    """
    input:
        datasets_perlabel=expand(rules.plot_confusion_matrix.output.performance_metrics_per_metadata, zip,
                                 dataset=["tabula_sapiens", "tabula_sapiens", "human_disease"],
                                 metadata_col=["celltype", "organ_tissue", "Disease_subtype"],
                                 model=["{model}", "{model}", "{model}"],
                                 normed=["raw", "raw", "raw"]  # normed/raw are the same :|
                                 ),

        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        per_class_examples_plot=ZERO_SHOT_MODEL_RESULTS / "performance_metrics.selected_classes_and_datasets.pdf",
    conda:
        "cellwhisperer"
    params:  # matching to dataset inputs
        label_cols=[
            "celltype",
            "organ_tissue",
            "Disease_subtype",
        ],
        selected_sample_lists=[
            ["kidney epithelial cell", "erythrocyte","plasma cell"],
            ["Kidney", "Lung", "Tongue"],
            ["Dilated cardiomyopathy","Melanoma","Hepatocellular carcinoma"],
        ],
        datasets=["tabula_sapiens", "tabula_sapiens", "human_disease"],
        suffix_prefix_dict=SUFFIX_PREFIX_DICT,
    resources:
        mem_mb=2000,
        slurm="cpus-per-task=1"
    log:
        notebook="../logs/zero_shot_performance_examples_{model}.ipynb"
    notebook:
        "../notebooks/zero_shot_performance_examples.py.ipynb"
