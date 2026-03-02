# SpotWhisperer evaluation pipeline for lung tissue datasets
SPOTWHISPERER_RESULTS = PROJECT_DIR / "results/spotwhisperer_eval/lung"
SPOTWHISPERER_MODEL_RESULTS = SPOTWHISPERER_RESULTS / "{model}"

# Lung tissue data paths
LUNG_DATA = PROJECT_DIR / "resources/lung_tissue"
LUNG_RESULTS = PROJECT_DIR / "results/lung_evaluation"
LUNG_MODEL_RESULTS = LUNG_RESULTS / "{model}"

# Lung dataset samples (from baseline CSV)
LUNG_SAMPLES = ["lc_1", "lc_2", "lc_3", "lc_4", "lc_5"]

LUNG_TISSUE_METADATA_COLS = ["region_type_expert_annotation", "cell_type_annotations"]
METRICS = ["accuracy", "f1", "auroc"]

rule zero_shot_lung_prediction:
    """
    Zero-shot property prediction for lung tissue datasets; outputs per-cell or per-class predictions.
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
    Plot confusion matrices and compute performance metrics for lung tissue predictions.
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


# rule compute_spotwhisperer_umap:
#     """
#     Compute UMAP embedding for SpotWhisperer datasets
#     TODO tbd, if needed
#     """
#     input:
#         processed_dataset=rules.process_full_dataset.output.model_outputs,
#         raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"]
#     output:
#         umap=SPOTWHISPERER_MODEL_RESULTS / "datasets" / "{dataset}" / "X_umap_on_neighbors_{model}.npz"
#     conda:
#         "cellwhisperer"
#     resources:
#         mem_mb=400000,
#         slurm="cpus-per-task=2"
#     log:
#         notebook="../logs/compute_spotwhisperer_umap_{dataset}_{model}.ipynb"
#     notebook:
#         "../notebooks/compute_spotwhisperer_umap.py.ipynb"


# rule plot_spotwhisperer_spatial:
#     """
#     Plot spatial distribution of predictions on tissue sections
#     TODO tbd, if needed
#     """
#     input:
#         predictions=rules.zero_shot_lung_prediction.output.predictions,
#         raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
#         umap=rules.compute_spotwhisperer_umap.output.umap,
#         mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
#     output:
#         spatial_plot=SPOTWHISPERER_MODEL_RESULTS / "plots" / "{dataset}" / "spatial_{metadata_col}_{grouping}.png",
#     conda:
#         "cellwhisperer"
#     resources:
#         mem_mb=200000,
#         slurm="cpus-per-task=2"
#     log:
#         notebook="../logs/plot_spotwhisperer_spatial_{model}_{dataset}_{metadata_col}_{grouping}.ipynb"
#     notebook:
#         "../notebooks/plot_spotwhisperer_spatial.py.ipynb"


# Performance summary across all lung tissue datasets
rule lung_performance_summary:
    """
    Aggregate lung tissue performance metrics into summary and plot.
    """
    input:
        performance_files=ancient(expand(  # I've put this on ancient so that it doesn't recompute the quilt1m evaluations (which would need model retraining)
            SPOTWHISPERER_MODEL_RESULTS / "performance" / "{dataset}" / "{metadata_col}_{grouping}_metrics.json",
            dataset=["lung_tissue"],
            metadata_col=LUNG_TISSUE_METADATA_COLS,
            grouping=["by_cell"],
            allow_missing=True
        ))
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
            model=[
                "spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m",
                "spotwhisperer_cellxgene_census__archs4_geo__hest1k",
            ],
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
            model=[
                "spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m",
                "spotwhisperer_cellxgene_census__archs4_geo__hest1k",
            ]
        )
    default_target: True  # TODO delete


# Baseline evaluation rules for lung tissue
rule lung_download_sample:
    """
    Download individual LC{N}.h5ad.gz files from the SpotWhisperer data server.
    Used as per-sample ground truth for baseline metric computation.
    """
    output:
        adata=LUNG_DATA / "per_sample" / "LC{sample_n}.h5ad",
    params:
        url="https://medical-epigenomics.org/papers/spotwhisperer/data/LC{sample_n}.h5ad.gz",
    wildcard_constraints:
        sample_n="[1-5]"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1 partition=cmackall"
    shell: """
        tmp=$(mktemp --suffix=.h5ad.gz)
        curl -fsSL {params.url} -o "$tmp"
        python -c "
import gzip, shutil, anndata
adata = anndata.read_h5ad('$tmp')
adata.write_h5ad('{output.adata}')
"
        rm -f "$tmp"
    """


rule lung_metrics_from_scores:
    """
    Aggregate per-class metrics across all lung samples from stored score CSVs and h5ad files.
    """
    input:
        scores=lambda wildcards: expand(
            LUNG_RESULTS / "{model}" / "{sample}_scores_seed0.csv",
            model=wildcards.model,
            sample=LUNG_SAMPLES,
        ),
        adatas=expand(
            LUNG_DATA / "per_sample" / "LC{sample_n}.h5ad",
            sample_n=[1, 2, 3, 4, 5],
        ),
    output:
        aggregated=LUNG_RESULTS / "{model}" / "summary" / "metrics_aggregated.json",
        per_class=LUNG_RESULTS / "{model}" / "summary" / "per_class_metrics.csv",
        per_dataset=LUNG_RESULTS / "{model}" / "summary" / "per_dataset_metrics.csv",
        per_class_by_dataset=LUNG_RESULTS / "{model}" / "summary" / "per_class_by_dataset_metrics.csv",
    wildcard_constraints:
        model="[^/]+",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1 partition=cmackall"
    script:
        "../scripts/compute_lung_metrics_from_scores.py"


rule lung_split_baseline_scores:
    """
    Split baseline logits (terms1) into a per-sample score CSV.
    Uses wildcards for baseline (conch|plip), sample, and seed.
    """
    input:
        baseline_csv=LUNG_DATA / "baselines_animesh_computed/{baseline}_logits_{terms_id}.csv",
    output:
        score=LUNG_RESULTS / "{baseline}_{terms_id}" / "{sample}_scores_seed{seed}.csv",
    wildcard_constraints:
        baseline="(conch|plip)",
        seed="0",
        sample="lc_[0-9]+",
        terms_id="(terms1|terms2)"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=1 partition=cmackall"
    script:
        "../scripts/split_baseline_logits_lung.py"

ruleorder: lung_split_baseline_scores > zero_shot_lung_prediction


rule lung_baselines_vs_spotwhisperer:
    """
    Per-class metrics comparison: SpotWhisperer models vs PLIP/CONCH baselines on lung tissue.
    Analogous to pathocell_baselines_vs_trimodal.
    SpotWhisperer predictions come from zero_shot_lung_prediction (LC1, 2 classes: TUM/TLS).
    Baseline per-class CSVs come from lung_metrics_from_scores (5 samples, 4 classes).
    Comparison is restricted to shared classes: tumor cells and tertiary lymphoid structure.
    """
    input:
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        # SpotWhisperer prediction CSVs (region_type_expert_annotation.by_cell.csv)
        sw_bibridge=SPOTWHISPERER_RESULTS / "spotwhisperer_cellxgene_census__archs4_geo__hest1k" / "datasets" / "lung_tissue" / "predictions" / "region_type_expert_annotation.by_cell.csv",
        sw_trimodal=SPOTWHISPERER_RESULTS / "spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m" / "datasets" / "lung_tissue" / "predictions" / "region_type_expert_annotation.by_cell.csv",
        # Baseline per-class summary CSVs from lung_metrics_from_scores
        baseline_conch_terms1=LUNG_RESULTS / "conch_terms1" / "summary" / "per_class_metrics.csv",
        baseline_plip_terms1=LUNG_RESULTS / "plip_terms1" / "summary" / "per_class_metrics.csv",
    output:
        plot=LUNG_RESULTS / "comparison" / "plots" / "per_class__{metric}__baselines_vs_spotwhisperer.svg",
        csv_table=LUNG_RESULTS / "comparison" / "tables" / "per_class_{metric}__baselines_vs_spotwhisperer.csv",
    params:
        metric="{metric}",
    wildcard_constraints:
        metric="(f1|rocauc|precision|accuracy)",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=8000,
        slurm="cpus-per-task=1 partition=cmackall"
    script:
        "../scripts/plot_lung_baselines_vs_spotwhisperer.py"


# NOTE: These rules are great, but not debugged yet :). Animesh ran them manually
# # Baseline runners using local scripts and project data structure
# rule lung_conch_baseline:
#     output:
#         logits_terms1=LUNG_DATA / "baselines_animesh_computed" / "conch_logits_terms1.csv",
#     params:
#         data_dir=LUNG_DATA / "processed",
#     conda:
#         "cellwhisperer"
#     resources:
#         mem_mb=32000,
#         slurm=slurm_gres("medium", num_cpus=4, time="4:00:00")
#     script:
#         "../scripts/run_conch_baseline_lung.py"

# rule lung_plip_baseline:
#     output:
#         logits_terms1=LUNG_DATA / "baselines_animesh_computed" / "plip_logits_terms1.csv",
#     params:
#         data_dir=LUNG_DATA / "processed",
#     conda:
#         "cellwhisperer"
#     resources:
#         mem_mb=32000,
#         slurm=slurm_gres("medium", num_cpus=4, time="4:00:00")
#     script:
#         "../scripts/run_plip_baseline_lung.py"
