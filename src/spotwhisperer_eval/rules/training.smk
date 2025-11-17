# Training rules for SpotWhisperer

rule train_spotwhisperer:
    """
    Train a SpotWhisperer model for a dataset_combo.
    Uses the base config and overrides dataset names; outputs a checkpoint.
    """
    input:
        base_config=ancient(PROJECT_DIR / "src/spotwhisperer_v3.yaml")
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
        mem_mb=250000,
        slurm=slurm_gres("large", num_cpus=20)
    shell: """
        cd {params.project_dir}
        cellwhisperer fit \
            --config {input.base_config} \
            --data.dataset_names {params.dataset_names} \
            --nproc 20 \
            {params.test_run_config} \
            --seed_everything {params.seed} \
            --last_model_path {output.model} \
            --wandb spotwhisperer_eval_{wildcards.dataset_combo}
    """
