from cellwhisperer.config import get_path
from snakemake.io import directory

# Extended fig 2

include: "../../shared/rules/training_sample_weights.smk"

# rule cw_transcriptome_term_scores:
#     """
#     - Compute the term-based match scores using (CW)
#         - Embed all terms (present in gsva_results)
#         - Use processed dataset and get cos_sim between each cell and each term
#     """
#     input:
#         processed_dataset=rules.process_full_dataset.output.model_outputs,  # PROJECT_DIR / config["paths"]["model_processed_dataset"],
#         gsva_results=PROJECT_DIR / config["paths"]["gsva"]["result"],
#         model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",  # needed for the keywords
#     output:
#         cw_transcriptome_term_scores=PROJECT_DIR / config["paths"]["gsva"]["cw_transcriptome_term_scores"],
#     conda:
#         "cellwhisperer"
#     resources:
#         mem_mb=40000,
#         slurm="cpus-per-task=5 gres=gpu:a100:1 qos=a100 partition=gpu"
#     log:
#         notebook="../logs/gsva_correlation_{model}_{dataset}.log"
#     notebook:
#         "../notebooks/cw_transcriptome_term_scores.py.ipynb"


# rule plot_gsva_correlations:
#     """
#     """
#     input:
#         cw_transcriptome_term_scores=config["paths"]["gsva"]["cw_transcriptome_term_scores"],
#         gsva_results=PROJECT_DIR / config["paths"]["gsva"]["result"],
#         mpl_style=ancient(PROJECT_DIR / config["plot_style"])
#     output:
#         gsva_correlation_results=PROJECT_DIR / config["paths"]["gsva"]["correlation"],
#         top_term_correlation=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_top_term_correlation.svg",
#         library_correlations=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_library_correlations.svg",
#         term_level_correlations=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_term_level_correlations.svg",
#         omim_correlations=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_omim_correlations.svg",
#         cw_binarized_gsva_scores=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "cw_binarized_gsva_scores.svg",
#         library_ks_statistics=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_library_ks_statistics.svg",  # also binarized
#         cherry_picked_examples=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_cherry_picked_examples.svg",
#     params:
#         selected_top_term="Pluripotent Stem Cells"  # not the one we want..
#     conda:
#         "cellwhisperer"
#     resources:
#         mem_mb=200000,
#         slurm="cpus-per-task=2"
#     log:
#         notebook="../logs/plot_gsva_correlation_{dataset}_{model}.ipynb"
#     notebook:
#         "../notebooks/plot_gsva_correlations.py.ipynb"

rule plot_zero_shot_validations:
    """
    """
    input:
        raw_read_count_tables =  expand(get_path(["paths", "read_count_table"],dataset="{dataset}"), dataset=TARGET_DATASETS),
    output:
        tabsap_wellstudied_celltypes_on_umap=get_path(["paths", "zero_shot_validation","result_dir"], model="{model}")/"tabula_sapiens/cellwhisperer_predictions.celltype_as_label.X_umap_on_neighbors_cellwhisperer.celltype.pdf",
        tabsap_wellstudied_predictions_on_umap=get_path(["paths", "zero_shot_validation","result_dir"], model="{model}")/"tabula_sapiens/cellwhisperer_predictions.celltype_as_label.X_umap_on_neighbors_cellwhisperer.predicted_labels_cellwhisperer.pdf",
        tabsap_erythrocytes_on_umap=get_path(["paths", "zero_shot_validation","result_dir"], model="{model}")/"tabula_sapiens/umap_on_neighbors_cellwhisperer.keyword_erythrocyte.asymmetrical_cmap.pdf",
        tabsap_all_celltypes_on_umap=get_path(["paths", "zero_shot_validation","result_dir"], model="{model}")/"tabula_sapiens/embedding_plots_MS_zero_shot.pdf",
        tabsap_integration_scores=get_path(["paths", "zero_shot_validation","result_dir"], model="{model}")/"tabula_sapiens/integration_scores.pdf",
        tabsap_wellstudied_confusion_matrix=get_path(["paths", "zero_shot_validation","result_dir"], model="{model}")/"tabula_sapiens_well_studied_celltypes/confusion_matrix_cellwhisperer.celltype_as_label.norm_True.pdf",
        rocauc_accuracy_overview=get_path(["paths", "zero_shot_validation","result_dir"], model="{model}")/"performance_metrics_cellwhisperer.selected_datasets.rocauc_and_accuracy.pdf",
        rocauc_accuracy_examples=get_path(["paths", "zero_shot_validation","result_dir"], model="{model}")/"performance_metrics_cellwhisperer.selected_classes_and_datasets.pdf",
    params:
        datasets = TARGET_DATASETS,
        model = "{model}"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=200000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/plot_zero_shot_validations{dataset}_{model}.ipynb"
    notebook:
        "../notebooks/zero_shot/plot_zero_shot_validations.py.ipynb"