rule process_full_dataset:
    """
    Run CellWhisperer on the read_count_table (provided) dataset and store the outputs in an npz file.
    """
    input:
        read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        model=ancient(PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt"),
    output:
        model_outputs=protected(PROJECT_DIR / config["paths"]["model_processed_dataset"]),
    resources:
        mem_mb=600000,  # could be made more efficient...
        slurm="cpus-per-task=5 gres=gpu:a100:1 qos=a100 partition=gpu"
    log:
        notebook="../logs/notebooks/process_full_dataset_{dataset}_{model}.py.ipynb",
        log_file="../logs/process_full_dataset_{dataset}_{model}.log"
    conda:
        "cellwhisperer"
    notebook:
        "../notebooks/process_full_dataset.py.ipynb"
