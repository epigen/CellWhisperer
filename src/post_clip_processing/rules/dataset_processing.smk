rule process_full_dataset:
    """
    Run single ceLLM on the full (provided) dataset and store all outputs in a single file (features, embeddings, and cross-modal-similarities).

    By setting `min_genes=1` in the data module, we try to include all cells in the dataset. Need to double check whether this worked via the log file.
    """
    input:
        full_data=PROJECT_DIR / config["paths"]["full_dataset"],
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / f"{CLIP_MODEL}.ckpt",
    output:
        model_outputs=PROJECT_DIR / config["paths"]["model_processed_dataset"].format(model=CLIP_MODEL, dataset="{dataset}"),
    resources:
        mem_mb=10000
    log:
        notebook="../logs/notebooks/process_full_dataset_{dataset}.py.ipynb",
        log_file="../logs/process_full_dataset_{dataset}.log"
    conda:
        "single-cellm"
    notebook:
        "../notebooks/process_full_dataset.py.ipynb"

