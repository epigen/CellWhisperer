# PathoCellBench evaluation pipeline for cell type classification
# This pipeline downloads PathoCell dataset, processes it into CellWhisperer format,
# and evaluates cell type prediction performance

from pathlib import Path as _Path

# TODO all processed files should use PATHOCELL_RESULTS (currently most use PATHOCELL_DATA)
PATHOCELL_RESULTS = PROJECT_DIR / "results/pathocell_evaluation"
PATHOCELL_DATA = PROJECT_DIR / "resources/pathocell"  # this is for downloaded files only
PATHOCELL_MODEL_RESULTS = PATHOCELL_RESULTS / "{model}"

# _hdf_dir = PATHOCELL_DATA / "raw/pathocell_hdf"
# DATASETS = sorted([p.stem for p in _hdf_dir.glob("*.hdf")])
DATASETS = ["reg006_B", "reg014_B", "reg022_B", "reg030_A", "reg037_B", "reg046_A", "reg056_A", "reg007_A", "reg015_A", "reg023_A", "reg030_B", "reg038_A", "reg047_A", "reg058_A", "reg007_B", "reg016_A", "reg023_B", "reg031_A", "reg039_A", "reg048_A", "reg059_A", "reg001_A", "reg008_A", "reg016_B", "reg024_B", "reg031_B", "reg039_B", "reg048_B", "reg059_B", "reg001_B", "reg008_B", "reg017_A", "reg025_A", "reg032_A", "reg040_A", "reg049_A", "reg060_A", "reg002_A", "reg009_A", "reg017_B", "reg025_B", "reg032_B", "reg040_B", "reg050_A", "reg060_B", "reg002_B", "reg009_B", "reg018_A", "reg026_A", "reg033_A", "reg041_A", "reg050_B", "reg061_A", "reg003_A", "reg010_A", "reg018_B", "reg026_B", "reg033_B", "reg041_B", "reg051_A", "reg062_A", "reg003_B", "reg010_B", "reg019_A", "reg027_A", "reg034_A", "reg042_A", "reg051_B", "reg063_A", "reg004_A", "reg011_A", "reg020_A", "reg027_B", "reg035_A", "reg042_B", "reg052_A", "reg064_A", "reg004_B", "reg011_B", "reg020_B", "reg028_A", "reg035_B", "reg043_A", "reg052_B", "reg065_A", "reg005_A", "reg012_A", "reg021_A", "reg028_B", "reg036_A", "reg044_A", "reg053_A", "reg066_A", "reg005_B", "reg012_B", "reg021_B", "reg029_A", "reg036_B", "reg045_A", "reg054_A", "reg067_A", "reg006_A", "reg013_B", "reg022_A", "reg029_B", "reg037_A", "reg045_B", "reg055_A", "reg068_A"]


rule pathocell_download_dataset:
    """
    Download PathoCell dataset from HuggingFace.
    Downloads the HDF5 format which contains images, masks, and cell type annotations.
    """
    output:
        dataset_marker=touch(PATHOCELL_DATA / "download_complete.marker"),
        data_dir=directory(PATHOCELL_DATA / "raw")
    conda:
        "cellwhisperer"
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=2"
    log:
        "logs/pathocell_download_dataset.log"
    shell: """
        LOG=$(realpath {log})
        mkdir -p {output.data_dir}
        
        # Download PathoCell dataset using huggingface-cli
        echo "Downloading PathoCell dataset from HuggingFace..." > $LOG
        
        # Download the dataset files
        huggingface-cli download \
            Kainmueller-Lab/PathoCell \
            --repo-type dataset \
            --local-dir {output.data_dir} \
            --local-dir-use-symlinks False \
            2>&1 | tee -a $LOG
        
        echo "Download complete" >> $LOG
    """


rule pathocell_process_data:
    """
    Process PathoCell data into CellWhisperer format.
    Converts a PathoCell HDF file to AnnData with spatial coordinates.
    """
    input:
        dataset_marker=rules.pathocell_download_dataset.output.dataset_marker,
        data_dir=rules.pathocell_download_dataset.output.data_dir,
        ct_mapping_fine=lambda wildcards: PATHOCELL_DATA / "raw/pathocell_hdf/CT_mapping.txt",
        ct_mapping_coarse=lambda wildcards: PATHOCELL_DATA / "raw/pathocell_hdf/CT_coarse_mapping.txt",
        hdf_file=lambda wildcards: PATHOCELL_DATA / f"raw/pathocell_hdf/{wildcards.dataset}.hdf"
    output:
        adata=PATHOCELL_DATA / "processed/{dataset}_{prediction_level}.h5ad",
        image=PATHOCELL_DATA / "processed/{dataset}_{prediction_level}.tiff",
        metadata=PATHOCELL_DATA / "processed/{dataset}_{prediction_level}_metadata.json"
    conda:
        "cellwhisperer"
    params:
        patch_level=lambda wildcards: wildcards.prediction_level=="patch"
    wildcard_constraints:
        prediction_level="(cell|patch)",
        dataset="[^/]+"
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=2 partition=cmackall"
    log:
        notebook="logs/pathocell_process_data_{dataset}_{prediction_level}.ipynb"
    notebook:
        "../notebooks/pathocell_process_data.py.ipynb"


rule pathocell_cell_type_prediction:
    """
    Run cell type prediction and evaluation using CellWhisperer model.
    Uses score_left_vs_right() or get_performance_metrics_left_vs_right()
    for evaluation. Supports both cell-level and patch-level prediction.
    """
    input:
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
        adata=rules.pathocell_process_data.output.adata,
        image=rules.pathocell_process_data.output.image
    output:
        results=PATHOCELL_MODEL_RESULTS / "{dataset}_{prediction_level}_prediction_seed{seed}.json",
        per_class_metrics=PATHOCELL_MODEL_RESULTS / "{dataset}_{prediction_level}_per_class_seed{seed}.csv",
        confusion_matrix=PATHOCELL_MODEL_RESULTS / "{dataset}_{prediction_level}_confusion_seed{seed}.csv"
    params:
        prediction_level="{prediction_level}",
    threads: 8
    wildcard_constraints:
        prediction_level="(cell|patch)",
        dataset="[^/]+",
        seed="\\d+"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=50000,
        slurm=slurm_gres("medium", num_cpus=8, time="2:00:00")
    log:
        notebook="logs/pathocell_cell_type_prediction_{model}_{dataset}_{prediction_level}_seed{seed}.ipynb"
    notebook:
        "../notebooks/pathocell_cell_type_prediction.py.ipynb"


rule pathocell_aggregate_results:
    """
    Aggregate results across multiple seeds for a model.
    """
    input:
        results=lambda wildcards: expand(
            rules.pathocell_cell_type_prediction.output.results,
            model=wildcards.model,
            prediction_level=wildcards.prediction_level,
            dataset=DATASETS,
            seed=SEEDS
        ),
    output:
        summary=PATHOCELL_MODEL_RESULTS / "summary" / "{prediction_level}_classification_summary.json"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1"
    log:
        notebook="logs/pathocell_aggregate_results_{model}_{prediction_level}.ipynb"
    notebook:
        "../notebooks/pathocell_aggregate_results.py.ipynb"


rule aggregate_pathocell_results:
    """
    Copy aggregated PathoCellBench summaries into the benchmarks directory for dataset_combo.
    This rule is used by the spider plot to access PathoCellBench results.

    """
    input:
        performance_summary=lambda wildcards: expand(
            rules.pathocell_aggregate_results.output.summary,
            model="spotwhisperer_{}".format(wildcards.dataset_combo),
            prediction_level="cell",  # Default to cell-level for backwards compatibility
            allow_missing=True,
        )
    output:
        aggregated_pathocell=BENCHMARKS_DIR / "pathocell" / "{dataset_combo}" / "performance_summary.json"
    wildcard_constraints:
        dataset_combo="[^/]+"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1"
    shell: """
        mkdir -p $(dirname {output.aggregated_pathocell})
        cp {input.performance_summary} {output.aggregated_pathocell}
    """


# Main rule to run all PathoCellBench evaluation
rule pathocell_compare_models:
    """
    Statistical comparison between two models based on aggregated *_values and per-class metrics.
    Performs t-tests and effect sizes; outputs CSVs and a summary JSON.
    """
    input:
        model_a_summary=lambda wildcards: PATHOCELL_RESULTS / f"{wildcards.model_a}/summary/{wildcards.prediction_level}_classification_summary.json",
        model_b_summary=lambda wildcards: PATHOCELL_RESULTS / f"{wildcards.model_b}/summary/{wildcards.prediction_level}_classification_summary.json",
        per_class_a=lambda wildcards: expand(
            PATHOCELL_RESULTS / "{model}" / "{dataset}_{prediction_level}_per_class_seed{seed}.csv",
            model=wildcards.model_a,
            prediction_level=wildcards.prediction_level,
            dataset=DATASETS,
            seed=SEEDS,
            allow_missing=True,
        ),
        per_class_b=lambda wildcards: expand(
            PATHOCELL_RESULTS / "{model}" / "{dataset}_{prediction_level}_per_class_seed{seed}.csv",
            model=wildcards.model_b,
            prediction_level=wildcards.prediction_level,
            dataset=DATASETS,
            seed=SEEDS,
            allow_missing=True,
        ),
    output:
        metrics_comparison_csv=PATHOCELL_RESULTS / "comparison" / "{prediction_level}" / "{model_a}_vs_{model_b}_metrics.csv",
        per_class_comparison_csv=PATHOCELL_RESULTS / "comparison" / "{prediction_level}" / "{model_a}_vs_{model_b}_per_class.csv",
        summary_json=PATHOCELL_RESULTS / "comparison" / "{prediction_level}" / "{model_a}_vs_{model_b}_summary.json",
    params:
        model_a="{model_a}",
        model_b="{model_b}",
        prediction_level="{prediction_level}",
    wildcard_constraints:
        prediction_level="(cell|patch)",
        model_a="[^/]+",
        model_b="[^/]+",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/compare_pathocell_models.py"

rule pathocell_per_class:
    """
    Create per-class plots of delta-score (model_a - model_b) for a given metric.
    """
    input:
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        per_class_a=lambda wildcards: expand(
            PATHOCELL_RESULTS / "{model}" / "{dataset}_{prediction_level}_per_class_seed{seed}.csv",
            model=wildcards.model_a,
            prediction_level=wildcards.prediction_level,
            dataset=DATASETS,
            seed=SEEDS,
            allow_missing=True,
        ),
        per_class_b=lambda wildcards: expand(
            PATHOCELL_RESULTS / "{model}" / "{dataset}_{prediction_level}_per_class_seed{seed}.csv",
            model=wildcards.model_b,
            prediction_level=wildcards.prediction_level,
            dataset=DATASETS,
            seed=SEEDS,
            allow_missing=True,
        ),
    output:
        plot=PATHOCELL_RESULTS / "comparison" / "{prediction_level}" / "plots" / "per_class__{metric}__{model_a}_vs_{model_b}.svg",
    params:
        model_a="{model_a}",
        model_b="{model_b}",
        prediction_level="{prediction_level}",
        metric="{metric}",
    wildcard_constraints:
        prediction_level="(cell|patch)",
        model_a="[^/]+",
        model_b="[^/]+",
        # Limit metric to avoid overlap with scatter outputs (no underscores)
        metric="[A-Za-z0-9@]+",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=8000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/plot_pathocell_perclass.py"

rule pathocell_per_class_f1_scatter:
    """
    Scatterplot of per-class mean F1 between two models.
    """
    input:
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        per_class_a=lambda wildcards: expand(
            PATHOCELL_RESULTS / "{model}" / "{dataset}_{prediction_level}_per_class_seed{seed}.csv",
            model=wildcards.model_a,
            prediction_level=wildcards.prediction_level,
            dataset=DATASETS,
            seed=SEEDS,
            allow_missing=True,
        ),
        per_class_b=lambda wildcards: expand(
            PATHOCELL_RESULTS / "{model}" / "{dataset}_{prediction_level}_per_class_seed{seed}.csv",
            model=wildcards.model_b,
            prediction_level=wildcards.prediction_level,
            dataset=DATASETS,
            seed=SEEDS,
            allow_missing=True,
        ),
    output:
        plot=PATHOCELL_RESULTS / "comparison" / "{prediction_level}" / "plots" / "f1_{model_a}_vs_{model_b}.svg",
    params:
        model_a="{model_a}",
        model_b="{model_b}",
        prediction_level="{prediction_level}",
        plot_type="violin",
        scatter_unit="dataset"  # 'class' or 'dataset'
    wildcard_constraints:
        prediction_level="(cell|patch)",
        model_a="[^/]+",
        model_b="[^/]+",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=6000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/plot_pathocell_f1_scatter.py"


rule pathocell_all:
    """
    Run complete PathoCellBench evaluation for cell type classification.
    """
    input:
        # Aggregate results per model and prediction level across datasets
        # expand(
        #     rules.pathocell_aggregate_results.output.summary,
        #     model=["spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m", "spotwhisperer_quilt1m"],
        #     prediction_level=["patch"]  # "cell", 
        # ),
        # # Statistical comparison between the two models
        # expand(
        #     rules.pathocell_compare_models.output.summary_json,
        #     model_a=["spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m"],
        #     model_b=["spotwhisperer_quilt1m"],
        #     prediction_level=["patch"],
        # ),
        # expand(
        #     rules.pathocell_compare_models.output.metrics_comparison_csv,
        #     model_a=["spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m"],
        #     model_b=["spotwhisperer_quilt1m"],
        #     prediction_level=["patch"],
        # ),
        # expand(
        #     rules.pathocell_compare_models.output.per_class_comparison_csv,
        #     model_a=["spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m"],
        #     model_b=["spotwhisperer_quilt1m"],
        #     prediction_level=["patch"],
        # ),
        # Per-class plots for key metrics
        expand(
            rules.pathocell_per_class.output.plot,
            model_a=["spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m"],
            model_b=["spotwhisperer_quilt1m"],
            prediction_level=["patch"],
            metric=["auroc", "f1", "accuracy", "precision", "recall@5"],
        ),
        # Per-class F1 scatter
        expand(
            rules.pathocell_per_class_f1_scatter.output.plot,
            model_a=["spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m"],
            model_b=["spotwhisperer_quilt1m"],
            prediction_level=["patch"],
        )
    default_target: True
