# SpotWhisperer evaluation pipeline for lung tissue datasets
SPOTWHISPERER_RESULTS = PROJECT_DIR / "results/spotwhisperer_eval/lung"
SPOTWHISPERER_MODEL_RESULTS = SPOTWHISPERER_RESULTS / "{model}"

LUNG_TISSUE_METADATA_COLS = ["region_type_expert_annotation", "cell_type_annotations"]
METRICS = ["accuracy", "f1", "auroc"]

rule zero_shot_lung_prediction:
    """
    Zero-shot property prediction with CellWhisperer for lung tissue datasets

    Similar to zero_shot_cellwhisperer_prediction but adapted for spatial transcriptomics data

    NOTE: region-mapping (e.g. NOR -> normal cells) and cell type prefixing ("Sample of {cell type}") is done in the notebook

    NOTE: it would be fair actually to combine infiltrating cells and tls into "immune cells". Furthermore "normal cells" are not really a tissue. best would be to search for "T cells" and "B cells"
    """
    input:
        processed_dataset=rules.process_full_dataset.output.model_outputs,
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
    output:
        predictions=SPOTWHISPERER_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "predictions" / "{metadata_col}.{grouping,by_cell|by_class}.csv",
    params:
        use_prefix_suffix_version=True,
        average_by_class=lambda wildcards: wildcards.grouping == "by_class",
        filter_classes=["UNASSIGNED", "NOR", "INFL"],  # "normal cells" is not a thing and "infiltrating immune cells" overlaps with TLS (TLS can be seen as subset of infiltrating immune cells)
    wildcard_constraints:
        dataset="lung_tissue",
        metadata_col="cell_type_annotations|region_type_expert_annotation"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=35000,
        slurm=slurm_gres("small"),
    log:
        notebook="../logs/zero_shot_spotwhisperer_prediction_{model}_{dataset}_{metadata_col}_{grouping}.ipynb"
    notebook:
        "../notebooks/zero_shot_spotwhisperer_prediction.py.ipynb"


rule plot_spotwhisperer_confusion_matrix:
    """
    Plot confusion matrices for SpotWhisperer predictions
    """
    input:
        predictions=rules.zero_shot_lung_prediction.output.predictions,
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
    output:
        confusion_matrix=SPOTWHISPERER_MODEL_RESULTS / "plots" / "{dataset}" / "confusion_matrix_{metadata_col}_{grouping}.png",
        performance_metrics=SPOTWHISPERER_MODEL_RESULTS / "performance" / "{dataset}" / "{metadata_col}_{grouping}_metrics.json",
    wildcard_constraints:
        metadata_col=".*annotations?"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=100000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/plot_spotwhisperer_confusion_matrix_{model}_{dataset}_{metadata_col}_{grouping}.ipynb"
    notebook:
        "../notebooks/plot_spotwhisperer_confusion_matrix.py.ipynb"


rule compute_spotwhisperer_umap:
    """
    Compute UMAP embedding for SpotWhisperer datasets
    TODO tbd, if needed
    """
    input:
        processed_dataset=rules.process_full_dataset.output.model_outputs,
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"]
    output:
        umap=SPOTWHISPERER_MODEL_RESULTS / "datasets" / "{dataset}" / "X_umap_on_neighbors_{model}.npz"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=400000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/compute_spotwhisperer_umap_{dataset}_{model}.ipynb"
    notebook:
        "../notebooks/compute_spotwhisperer_umap.py.ipynb"


rule plot_spotwhisperer_spatial:
    """
    Plot spatial distribution of predictions on tissue sections
    TODO tbd, if needed
    """
    input:
        predictions=rules.zero_shot_lung_prediction.output.predictions,
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        umap=rules.compute_spotwhisperer_umap.output.umap,
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
    output:
        spatial_plot=SPOTWHISPERER_MODEL_RESULTS / "plots" / "{dataset}" / "spatial_{metadata_col}_{grouping}.png",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=200000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/plot_spotwhisperer_spatial_{model}_{dataset}_{metadata_col}_{grouping}.ipynb"
    notebook:
        "../notebooks/plot_spotwhisperer_spatial.py.ipynb"


# Performance summary across all lung tissue datasets
rule lung_performance_summary:
    """
    Compute and aggregate performance metrics
    """
    input:
        performance_files=expand(
            SPOTWHISPERER_MODEL_RESULTS / "performance" / "{dataset}" / "{metadata_col}_{grouping}_metrics.json",
            dataset=["lung_tissue"],
            metadata_col=LUNG_TISSUE_METADATA_COLS,
            grouping=["by_cell"],
            allow_missing=True
        )
    output:
        summary=SPOTWHISPERER_RESULTS / "performance_summary_{model}.csv",
        plot=SPOTWHISPERER_RESULTS / "performance_summary_{model}.png",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=1"
    log:
        notebook="../logs/spotwhisperer_performance_summary_{model}.ipynb"
    notebook:
        "../notebooks/spotwhisperer_performance_summary.py.ipynb"


# Main rule to generate all SpotWhisperer results
rule spotwhisperer_all:
    input:
        # Basic predictions
        expand(
            rules.zero_shot_lung_prediction.output.predictions,
            model=[config["model_name_path_map"]["spotwhisperer"]],
            dataset=["lung_tissue"],
            metadata_col=LUNG_TISSUE_METADATA_COLS,
            grouping=["by_cell"]
        ),
        # # Confusion matrices and performance metrics
        # expand(
        #     rules.plot_spotwhisperer_confusion_matrix.output.confusion_matrix,
        #     model=[config["model_name_path_map"]["spotwhisperer"]],
        #     dataset=LUNG_TISSUE_DATASETS,
        #     metadata_col=LUNG_TISSUE_METADATA_COLS,
        #     grouping=["by_cell", "by_class"]
        # ),
        # # Spatial plots
        # expand(
        #     rules.plot_spotwhisperer_spatial.output.spatial_plot,
        #     model=[config["model_name_path_map"]["spotwhisperer"]],
        #     dataset=LUNG_TISSUE_DATASETS,
        #     metadata_col=LUNG_TISSUE_METADATA_COLS,
        #     grouping=["by_cell", "by_class"]
        # ),
        # # Performance summary
        expand(
            rules.lung_performance_summary.output.summary,
            model=[config["model_name_path_map"]["spotwhisperer"]]
        )
    default_target: True  # TODO delete
