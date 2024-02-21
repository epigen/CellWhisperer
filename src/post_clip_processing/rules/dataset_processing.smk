rule integrate_dataset:
    """
    Note: this rule is copied from /home/moritz/Projects/cellwhisperer/src/annotation_processing/pipeline/Snakefile
    """
    input:
        read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        processed_annotations=PROJECT_DIR / config["paths"]["processed_annotations"]
    output:
        PROJECT_DIR / config["paths"]["full_dataset"]
    params:
        anndata_label_name=config["anndata_label_name"],
    conda:
        "cellwhisperer"
        # PROJECT_DIR / "envs" / "main.yaml"
    script:
        "../../annotation_processing/pipeline/scripts/integrate_dataset.py"


rule process_full_dataset:
    """
    Run CellWhisperer on the full (provided) dataset and store all outputs in a single file (features, embeddings, and cross-modal-similarities).

    By setting `min_genes=1` in the data module, we try to include all cells in the dataset. Need to double check whether this worked via the log file.
    """
    input:
        full_data=PROJECT_DIR / config["paths"]["full_dataset"],
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
    output:
        model_outputs=protected(PROJECT_DIR / config["paths"]["model_processed_dataset"]),  # TODO protection did not work :/
    resources:
        mem_mb=100000,
        slurm="cpus-per-task=5 gres=gpu:a100:1 qos=a100 partition=gpu"
    log:
        notebook="../logs/notebooks/process_full_dataset_{dataset}_{model}.py.ipynb",
        log_file="../logs/process_full_dataset_{dataset}_{model}.log"
    conda:
        "cellwhisperer"
    notebook:
        "../notebooks/process_full_dataset.py.ipynb"

