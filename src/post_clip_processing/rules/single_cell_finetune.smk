# DEPRECATED (we just train on massive amounts of single cells now)
FINETUNE_DATASET = "tabula_sapiens"
rule tabsap_finetuning:
    """
    TODO Things to test here
    - use of ckpt_path vs model_ckpt (the latter resets optimizer/trainer state and trains from scratch)
    - Use `val/clip_loss_epoch` with early stopping and train "as many epochs as it takes"
    - Note: The config can be retrieved from wandb using the `config` field.
      Also locally: results/wandb_logging/wandb/run-20240124_114902-coyxp1d6/files/config.yaml
    """
    input:
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
        full_data=PROJECT_DIR / config["paths"]["full_dataset"],
        config=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.yaml",
    output:
        PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}__{dataset}.ckpt",
    params:
        lr=1e-5,
        wandb_name="fine_tune_tabsap",
    resources:
        mem_mb=100000,
        slurm="cpus-per-task=20 gres=gpu:a100-sxm4-80gb:1 qos=a100-sxm4-80gb partition=gpu"
    conda:
        "cellwhisperer"
    shell:
        """
        # TODO note that this was last time run without the last 3 params
        cd /msc/home/mschae83/cellwhisperer/
        cellwhisperer fit \
        --config {input.config} \
        --model_ckpt {input.model} \
        --data.dataset_name {wildcards.dataset} \
        --best_model_path {output} \
        --wandb {params.wandb_name} \
        --model.learning_rate {params.lr} \
        --model.scheduler.init_args.T_max 100 \
        --trainer.max_epochs 100 \
        --model.lr_warmup_steps 100
        """
