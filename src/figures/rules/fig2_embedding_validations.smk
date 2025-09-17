
ZERO_SHOT_RESULTS = PROJECT_DIR / "results/plots/zero_shot_validation"
ZERO_SHOT_CW_MODEL_RESULTS = ZERO_SHOT_RESULTS / "{model,cellwhisperer_clip_.*}"
ZERO_SHOT_CW_MODEL_RESULTS_ESCAPED = ZERO_SHOT_RESULTS / "{{model,cellwhisperer_clip_.*}}" # The "*" causes issues otherwise

TRANSCRIPTOME_MODELS = config["scfms"] + CW_CLIP_MODELS
CELLTYPE_EVAL_DATASETS = ["tabula_sapiens", "pancreas", "immgen", "aida", "tabula_sapiens_well_studied_celltypes"]  # optionally: [d for d, cols in config["metadata_cols_per_zero_shot_validation_dataset"].items() if "celltype" in cols]
METRICS = ["accuracy", "f1", "auroc"]

from notebooks.zero_shot_validation_scripts.utils import SUFFIX_PREFIX_DICT  # TODO consider moving to config

include: "../../shared/rules/training_sample_weights.smk"
include: "prompt_sensitivity.smk"

# Computations
rule compute_umap_neighbors:
    """
    # TODO might need to move to another include
    """
    input:
        processed_dataset=PROJECT_DIR / config["paths"]["model_processed_dataset"],
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"]
    output:
        umap=ZERO_SHOT_RESULTS / "{model}" / "datasets" / "{dataset,[^/]+}" / "X_umap_on_neighbors_{model}.npz"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=400000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/compute_umap_neighbors_{dataset}_{model}.ipynb"
    notebook:
        "../notebooks/compute_umap_neighbors.py.ipynb"

include: "zero_shot_finetuning.smk"
include: "marker_based_celltypes.smk"

rule zero_shot_cellwhisperer_prediction:
    """
    Zero-shot property prediction with CellWhisperer

    For a given dataset and field, compute all predictions for downstream analysis

    NOTE: could be used as template for few-shot learing with other embedding-based model zero shot predictions
    """
    input:
        processed_dataset=PROJECT_DIR / config["paths"]["model_processed_dataset"],
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",  # needed to embed the keywords
    output:
        # Using a directory here because the exact files produced depend on the dataset:
        predictions=ZERO_SHOT_CW_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "predictions" / "{metadata_col}.{grouping,by_cell|by_class}.csv",  # NOTE: might be used as input for `plot_confusion_matrix_and_get_performance_metrics` as well (and by extension `zero_shot_performance_macroavg`)
        # scores=ZERO_SHOT_CW_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "predictions" / "{metadata_col}.{grouping}.scores.csv",  # TODO refactor away since everything is in predictions anyways
    params:
        use_prefix_suffix_version=True,
        average_by_class=lambda wildcards: wildcards.grouping == "by_class",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=350000,
        slurm=slurm_gres()
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
        processed_dataset=PROJECT_DIR / config["paths"]["model_processed_dataset"],
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        umap=rules.compute_umap_neighbors.output.umap,
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
    output:
        embedding_plots_zero_shot_comparison_pdf=ZERO_SHOT_RESULTS / "{model}" / "datasets" / "{dataset,[^/]+}" / "embedding_plots_zero_shot_comparison.pdf",
        embedding_plots_zero_shot_comparison_png=ZERO_SHOT_RESULTS / "{model}" / "datasets" / "{dataset,[^/]+}" / "embedding_plots_zero_shot_comparison.png",
        integration_scores=ZERO_SHOT_RESULTS / "{model}" / "datasets" / "{dataset,[^/]+}" / "embedding_scib_scores.json",
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
        [rules.transcriptome_embedding_scib.output.integration_scores.replace("{model}", model) for model in TRANSCRIPTOME_MODELS],
    output:
        integration_scores=ZERO_SHOT_RESULTS / "integration_scores_{dataset}.csv",
        integration_scores_plot=ZERO_SHOT_RESULTS / "integration_scores_{dataset}.pdf"
    conda:
        "cellwhisperer"
    params:
        models=TRANSCRIPTOME_MODELS
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
        processed_dataset=PROJECT_DIR / config["paths"]["model_processed_dataset"],  # PROJECT_DIR / config["paths"]["model_processed_dataset"],
        gsva_results=PROJECT_DIR / config["paths"]["gsva"]["result"],
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",  # needed for the keywords
    output:
        cw_transcriptome_term_scores=PROJECT_DIR / config["paths"]["gsva"]["cw_transcriptome_term_scores"],
    conda:
        "cellwhisperer"
    resources:
        mem_mb=40000,
        slurm=slurm_gres()
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
        processed_dataset=PROJECT_DIR / config["paths"]["model_processed_dataset"],
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        predictions=rules.zero_shot_cellwhisperer_prediction.output.predictions.replace("{grouping}", "by_cell"),
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        plot_dir=directory(ZERO_SHOT_CW_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "confidence_distribution_plots")  # NOTE: could be disentangled as well
    conda:
        "cellwhisperer"
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
        processed_dataset=PROJECT_DIR / config["paths"]["model_processed_dataset"],
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        predictions=rules.zero_shot_cellwhisperer_prediction.output.predictions.replace("{grouping}", "by_cell"),
        umap=rules.compute_umap_neighbors.output.umap,
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        result_dir=directory(ZERO_SHOT_CW_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "other_plots"),
    conda:
        "cellwhisperer"
    params:
        use_prefix_suffix_version=True
    resources:
        mem_mb=400000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/plot_zero_shot_predictions_on_umap_{model}_{dataset}_{metadata_col}.ipynb"
    notebook:
        "../notebooks/plot_zero_shot_predictions_on_umap.py.ipynb"


rule plot_confusion_matrix_and_get_performance_metrics:
    """
    NOTE: could probably use `scores` computed by `zero_shot_cellwhisperer_prediction`. This would work by refactoring the bloated function `get_performance_metrics_transcriptome_vs_text`, such that it takes as input only the scores and then computes the metrics.
    """
    input:
       # predictions=rules.zero_shot_cellwhisperer_prediction.output.predictions.replace("{grouping}", "by_cell"), # Not actually used
        processed_dataset=PROJECT_DIR / config["paths"]["model_processed_dataset"],
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
    output:
        confusion_matrix_plot=ZERO_SHOT_CW_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "confusionmatrix{normed}.pdf",
        confusion_matrix_table=ZERO_SHOT_CW_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "confusionmatrix{normed}.xlsx",
        performance_metrics=ZERO_SHOT_CW_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "performancemetrics{normed}.csv",  # note that this is the same for normed and unnormed (would be better to split the rule once more actually)
        performance_metrics_per_metadata=ZERO_SHOT_CW_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "{metadata_col}" / "performance_metrics_permetadata{normed}.csv",
    params:
        normed=lambda wildcards: wildcards.normed == "normed",
        use_prefix_suffix_version=True,
    resources:
        mem_mb=350000,
        slurm=slurm_gres()
    conda:
        "cellwhisperer"
    log:
        notebook="../logs/plot_confusion_matrix_and_get_performance_metrics_{model}_{dataset}_{metadata_col}_{normed}.ipynb"
    notebook:
        "../notebooks/plot_confusion_matrix_and_get_performance_metrics.py.ipynb"

rule plot_term_search_results:
    """
    Plot the ground truth celltype and the keyword search results on the UMAP (Fig 2a)
    """
    input:
        processed_dataset=lambda wildcards: PROJECT_DIR / config["paths"]["model_processed_dataset"].format(dataset="tabula_sapiens", model=wildcards.model),
        raw_read_count_table=str(PROJECT_DIR / config["paths"]["read_count_table"]).format(dataset="tabula_sapiens"),
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",  # needed to embed the keywords
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        umap=rules.compute_umap_neighbors.output.umap.replace("{dataset}", "tabula_sapiens")
    output:
        # Using a directory here because the exact files produced depend on the dataset:
        umap_on_neighbors_celltype=ZERO_SHOT_CW_MODEL_RESULTS / "datasets" / "tabula_sapiens" / "umap_on_neighbors_cellwhisperer.true_celltype_{celltype}.{file_suffix}",
        colorscale_symmetrical=ZERO_SHOT_CW_MODEL_RESULTS / "datasets" / "tabula_sapiens" / "umap_on_neighbors_cellwhisperer.keyword_for_{celltype}.symmetrical_cmap.{file_suffix}",
        colorscale_asymmetrical=ZERO_SHOT_CW_MODEL_RESULTS / "datasets" / "tabula_sapiens" / "umap_on_neighbors_cellwhisperer.keyword_for_{celltype}.asymmetrical_cmap.{file_suffix}",
    params:
        celltype_terms_dict=config["celltype_terms"],
        suffix_prefix_dict=SUFFIX_PREFIX_DICT,
        dataset = "tabula_sapiens",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=350000,
        slurm=slurm_gres()
    log:
        notebook="../logs/plot_term_search_results_{model}_{celltype}_tabula_sapiens_{file_suffix}.ipynb"
    notebook:
        "../notebooks/plot_term_search_results.py.ipynb"

rule cw_vs_basemodel_macroavg_comparisons:
    """
    Compare macroaverage celltype prediction performance of CellWhisperer (with a given transcriptome base model) vs. that basemodel (fine-tuned) 
    and vs a marker-based method (CellAssign)
    """
    input:
        cw_macroaverages=expand(rules.plot_confusion_matrix_and_get_performance_metrics.output.performance_metrics, zip,
                                 dataset=CELLTYPE_EVAL_DATASETS,
                                 metadata_col=["celltype"]*(len(CELLTYPE_EVAL_DATASETS)),
                                 normed=["raw"]*(len(CELLTYPE_EVAL_DATASETS)),  # normed/raw are the same :|
                                 allow_missing=True,
                                 ),
        aggregated_predictions_finetuned_models=expand(rules.aggregate_scfm_evaluations.output.aggregated_predictions,
                                 metric=METRICS,
                                 training_options=TRAINING_OPTIONS, ),
        marker_based_method_performances = expand(rules.cell_assign.output.performance, dataset=CELLTYPE_EVAL_DATASETS),
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        scores_across_training_options = expand(ZERO_SHOT_CW_MODEL_RESULTS_ESCAPED / "CW_vs_finetuned_base_model_macroavg_{metric}.csv",
                                 metric=METRICS),
        barplots_across_training_options = expand(ZERO_SHOT_CW_MODEL_RESULTS_ESCAPED / "CW_vs_finetuned_base_model_macroavg_{metric}.barplot.pdf",
                                    metric=METRICS),
        barplots_across_training_options_across_metrics = ZERO_SHOT_CW_MODEL_RESULTS / "CW_vs_finetuned_base_model_macroavg_all_metrics.barplot.pdf",
    conda:
        "cellwhisperer"
    params:
        label_cols=["celltype"]*len(CELLTYPE_EVAL_DATASETS),
        datasets=CELLTYPE_EVAL_DATASETS,
        metrics=METRICS,
        training_options=TRAINING_OPTIONS,
    resources:
        mem_mb=2000,
        slurm="cpus-per-task=1"
    log:
        notebook="../logs/cw_vs_basemodel_macroavg_comparisons_{model}.ipynb"
    notebook:
        "../notebooks/cw_vs_basemodel_macroavg_comparisons.py.ipynb"

rule zero_shot_performance_examples:
    """
    Example accuracies for zero-shot cell type property predictions
    """
    input:
        per_class_performance=expand(rules.plot_confusion_matrix_and_get_performance_metrics.output.performance_metrics_per_metadata, zip,
                                 dataset=[ "tabula_sapiens", "human_disease", "human_disease"],
                                 metadata_col=["organ_tissue", "Tissue","Disease_subtype",],
                                 model=["{model}", "{model}", "{model}"],
                                 normed=["raw", "raw", "raw"],  # normed/raw are the same :|,
                                 ),
        macroavg_performance=expand(rules.plot_confusion_matrix_and_get_performance_metrics.output.performance_metrics, zip,
                                 dataset=["tabula_sapiens", "human_disease", "human_disease"],
                                 metadata_col=["organ_tissue", "Tissue","Disease_subtype",],
                                 model=["{model}", "{model}", "{model}"],
                                 normed=["raw", "raw", "raw"],  # normed/raw are the same :|,
                                 ),
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        per_class_examples_plot=ZERO_SHOT_CW_MODEL_RESULTS / "performance_metrics.selected_classes_and_datasets.pdf",
    conda:
        "cellwhisperer"
    params:  # matching to dataset inputs
        label_cols=[
            "organ_tissue",
            "Tissue",
            "Disease_subtype",
        ],
        selected_sample_lists=[
            # ["Kidney", "Lung", "Tongue"],
            # ["Heart","Bowel","Liver"],
            # ["Dilated cardiomyopathy","Melanoma","Hepatocellular carcinoma"],
            ["Kidney", "Lung", "Tongue"],
            ["Heart","Ovary","Liver"],
            ["Dilated cardiomyopathy","Meningioma","Hepatocellular carcinoma"],
        ],
        datasets=["tabula_sapiens",  "human_disease","human_disease"],
        suffix_prefix_dict=SUFFIX_PREFIX_DICT,
    resources:
        mem_mb=2000,
        slurm="cpus-per-task=1"
    log:
        notebook="../logs/zero_shot_performance_examples_{model}.ipynb"
    notebook:
        "../notebooks/zero_shot_performance_examples.py.ipynb"

rule zero_shot_performance_suppl_table:
    """
    Create an xlsx file with one sheet per dataset, containing per-label performance values and the predicted classes (confusion matrix)
    """
    input:
        datasets_perlabel=expand(rules.plot_confusion_matrix_and_get_performance_metrics.output.performance_metrics_per_metadata, zip,
                                 dataset=["tabula_sapiens","tabula_sapiens", "tabula_sapiens_well_studied_celltypes", "pancreas", "immgen", "human_disease",  "human_disease", "aida"],
                                 metadata_col=["celltype","organ_tissue","celltype","celltype","celltype","Disease_subtype","Tissue","celltype"],
                                 model=["{model}"]*8,
                                 normed=["raw"]*8  # normed/raw are the same :|
                                 ),
    output:
        confusion_mtx_table=ZERO_SHOT_CW_MODEL_RESULTS / "performance_metrics_and_confusion_matrix_per_label.xlsx",
    conda:
        "cellwhisperer"
    params:
        label_cols=[
            "celltype",
            "organ_tissue",
            "celltype",
            "celltype",
            "celltype",
            "Disease_subtype",
            "Tissue",
            "celltype"],
        label_cols_pretty = [
            "Cell Type",
            "Organ,Tissue",
            "Cell Type",
            "Cell Type",
            "Cell Type",
            "Disease Subtype",
            "Tissue",
            "Cell Type"],
        datasets=[
            "tabula_sapiens",
            "tabula_sapiens",
            "tabula_sapiens_well_studied_celltypes",
            "pancreas",
            "immgen",
            "human_disease",
            "human_disease",
            "aida"],
        dataset_names_pretty = [
            "Tabula Sapiens",
            "Tabula Sapiens",
            "Tab. Sap. 20 common",
            "Pancreas",
            "ImmGen",
            "Human Disease",
            "Human Disease",
            "AIDA"],
    resources:
        mem_mb=2000,
        slurm="cpus-per-task=1"
    log:
        notebook="../logs/zero_shot_performance_create_xlsx_{model}.ipynb"
    notebook:
        "../notebooks/zero_shot_performance_create_xlsx.py.ipynb"

rule fig2_main:
    input:
        expand(rules.plot_confusion_matrix_and_get_performance_metrics.output.confusion_matrix_plot,
               model=CLIP_MODEL,
               dataset=["tabula_sapiens", "tabula_sapiens_well_studied_celltypes","aida","pancreas"],
               metadata_col="celltype",
               normed=["normed", "raw","normed","normed"]),
        expand(rules.zero_shot_performance_suppl_table.output.confusion_mtx_table, model=CLIP_MODEL),
        expand(rules.plot_term_search_results.output.umap_on_neighbors_celltype, celltype=config["celltype_terms"].keys(), file_suffix=["png"], model=CW_CLIP_MODELS),
        [
            base_fn.format(dataset=dataset, model=CLIP_MODEL, metadata_col=metadata_col)
            for base_fn in [rules.plot_zero_shot_predictions_on_umap.output.result_dir, rules.plot_confidence_distributions.output.plot_dir]
            for dataset in ["tabula_sapiens", "pancreas"]  # TODO config["metadata_cols_per_zero_shot_validation_dataset"].keys()
            for model in [CLIP_MODEL]
            for metadata_col in config["metadata_cols_per_zero_shot_validation_dataset"][dataset]
        ],

        expand(rules.cw_vs_basemodel_macroavg_comparisons.output.barplots_across_training_options_across_metrics, model=CW_CLIP_MODELS),
        expand(rules.zero_shot_performance_examples.output.per_class_examples_plot, model=CLIP_MODEL),
        expand(rules.aggregate_scfm_evaluations.output, training_options=TRAINING_OPTIONS, metric=["accuracy", "f1", "auroc"]),

        # Figure S2 (extended: embedding and gene set correlation analyses)
        expand(PROJECT_DIR / config["paths"]["gsva"]["correlation"], dataset=["human_disease"], model=CLIP_MODEL),  # NOTE: Can also be run for tabula_sapiens and for archs4_geo
        expand(rules.plot_joint_transcriptome_embedding_scib.output.integration_scores, dataset=["tabula_sapiens", "pancreas"]),

        # Marker-based analysis
        expand(
            rules.cell_assign.output.performance,
            dataset=CELLTYPE_EVAL_DATASETS,
        ),

        rules.plot_query_variant_cell_matching.output.plot.format(model=config["model_name_path_map"]["cellwhisperer_geneformer"]),
