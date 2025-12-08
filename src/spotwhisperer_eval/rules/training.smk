# Training rules for SpotWhisperer

rule subsample_dataset:
    """
    Create a randomly subsampled version of a dataset (1/nth of the data).
    Works on the h5ad dataset level for single-file datasets.
    """
    input:
        full_dataset=PROJECT_DIR / config["paths"]["full_dataset"]
    output:
        subsampled_dataset=PROJECT_DIR / "results/{dataset}_8thsub/full_data.h5ad"
    params:
        n=8,  # subsample to 1/8th
        seed=SEEDS[0]
    conda:
        "cellwhisperer"
    script:
        "../scripts/subsample_dataset.py"

rule subsample_multi_file_dataset:
    """
    Create a randomly subsampled version of a multi-file dataset (1/nth of the files).
    Works by symlinking a subset of h5ad files for datasets with multiple files.
    """
    input:
        h5ads_dir=PROJECT_DIR / "results/{dataset}/h5ads"
    output:
        subsampled_h5ads=directory(PROJECT_DIR / "results/{dataset}_8thsub/h5ads")
    params:
        n=8,  # subsample to 1/8th
        seed=SEEDS[0]
    conda:
        "cellwhisperer"
    script:
        "../scripts/subsample_multi_file_dataset.py"

rule train_spotwhisperer:
    """
    Train a SpotWhisperer model for a dataset_combo.
    Uses the base config and overrides dataset names; outputs a checkpoint.
    """
    input:
        base_config=ancient(BASE_CONFIG)
    output:
        model=protected(PROJECT_DIR / config["paths"]["jointemb_models"] / "spotwhisperer_{dataset_combo}.ckpt")
    params:
        dataset_names=lambda wildcards: wildcards.dataset_combo.replace("__", ","),
        test_run_config="--trainer.limit_train_batches 500 --trainer.max_epochs 2" if config.get("fast", False) else "",
        tmpmodel=PROJECT_DIR / config["paths"]["jointemb_models"] / "spotwhisperer_clip_v1.ckpt",
        seed=SEEDS[0],
        project_dir=PROJECT_DIR
    conda:
        "cellwhisperer"
    resources:
        mem_mb=150000,
        slurm=slurm_gres("large", num_cpus=12, time="48:00:00", num_gpus=1)
    shell: """
        cd {params.project_dir}
        cellwhisperer fit \
            --config {input.base_config} \
            --data.dataset_names {params.dataset_names} \
            {params.test_run_config} \
            --seed_everything {params.seed} \
            --last_model_path {output.model} \
            --omit_validation_functions \
            --wandb spotwhisperer_eval_{wildcards.dataset_combo}
    """
