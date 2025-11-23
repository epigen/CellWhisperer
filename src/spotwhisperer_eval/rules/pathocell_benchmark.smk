# PathoCellBench evaluation pipeline for cell type classification
# This pipeline downloads PathoCell dataset, processes it into CellWhisperer format,
# and evaluates cell type prediction performance

PATHOCELL_RESULTS = PROJECT_DIR / "results/pathocell_evaluation"
PATHOCELL_DATA = PROJECT_DIR / "resources/pathocell"
PATHOCELL_MODEL_RESULTS = PATHOCELL_RESULTS / "{model}"

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
    Converts the first TMA from HDF5 format to AnnData with spatial coordinates.
    Creates a dataset compatible with the lymphoma_cosmx_small format.
    """
    input:
        dataset_marker=rules.pathocell_download_dataset.output.dataset_marker,
        data_dir=rules.pathocell_download_dataset.output.data_dir
    output:
        adata=PATHOCELL_DATA / "processed/pathocell_tma1.h5ad",
        image=PATHOCELL_DATA / "processed/pathocell_tma1.svs",
        metadata=PATHOCELL_DATA / "processed/pathocell_tma1_metadata.json"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=4"
    log:
        notebook="logs/pathocell_process_data.ipynb"
    notebook:
        "../notebooks/pathocell_process_data.py.ipynb"


rule pathocell_cell_type_prediction:
    """
    Run cell type prediction and evaluation using CellWhisperer model.
    Uses score_left_vs_right() or get_performance_metrics_left_vs_right() 
    for evaluation.
    """
    input:
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
        adata=rules.pathocell_process_data.output.adata,
        image=rules.pathocell_process_data.output.image
    output:
        results=PATHOCELL_MODEL_RESULTS / "cell_type_prediction_seed{seed}.json",
        per_class_metrics=PATHOCELL_MODEL_RESULTS / "per_class_metrics_seed{seed}.csv",
        confusion_matrix=PATHOCELL_MODEL_RESULTS / "confusion_matrix_seed{seed}.csv"
    threads: 8
    wildcard_constraints:
        seed="\\d+"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=50000,
        slurm=slurm_gres("medium", num_cpus=8)
    log:
        notebook="logs/pathocell_cell_type_prediction_{model}_seed{seed}.ipynb"
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
            seed=SEEDS
        )
    output:
        summary=PATHOCELL_MODEL_RESULTS / "summary" / "cell_type_classification_summary.json"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1"
    log:
        notebook="logs/pathocell_aggregate_results_{model}.ipynb"
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
            allow_missing=True,
        )
    output:
        aggregated_pathocell=BENCHMARKS_DIR / "pathocell" / "{dataset_combo}" / "performance_summary.json"
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
rule pathocell_all:
    """
    Run complete PathoCellBench evaluation for cell type classification.
    """
    input:
        # Process the data
        rules.pathocell_process_data.output.adata,
        
        # Run predictions for key models
        expand(
            rules.pathocell_cell_type_prediction.output.results,
            model=["spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m"],
            seed=SEEDS[0]
        ),
        
        # Aggregate results
        expand(
            rules.pathocell_aggregate_results.output.summary,
            model=["spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m"]
        )
    default_target: True
