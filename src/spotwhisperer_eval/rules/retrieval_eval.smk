# Retrieval and zero-shot evaluation rules

rule spotwhisperer_test:
    """
    Evaluate a trained model on test_dataset via `cellwhisperer test`.
    Writes retrieval and zero-shot metrics to CSV logs.
    """
    input:
        model=rules.train_spotwhisperer.output.model,
        base_config=ancient(PROJECT_DIR / "src/spotwhisperer_v3.yaml")
    output:
        retrieval_and_cwevals=PROJECT_DIR / config["paths"]["csv_logs"] / "sweval___{dataset_combo}___{test_dataset}" / "metrics.csv",
        individual_clip_scores=PROJECT_DIR / config["paths"]["csv_logs"] / "sweval___{dataset_combo}___{test_dataset}" / "test_individual_clip_scores.csv",
    params:
        test_dataset=lambda wildcards: wildcards.test_dataset.replace("__", ","),
        seed=SEEDS[0],
        batch_size=128,
        project_dir=PROJECT_DIR,
    conda:
        "cellwhisperer"
    resources:
        mem_mb=100000,
        runtime=16*60,
        slurm=slurm_gres("small", num_cpus=4),
    shell: """
        cd {params.project_dir}
        cellwhisperer test \
            --config {input.base_config} \
            --model_ckpt {input.model} \
            --data.dataset_names {params.test_dataset} \
            --data.train_fraction 0.0 \
            --batch_size {params.batch_size} \
            --seed_everything {params.seed} \
            --nproc 4 \
            --trainer.logger.init_args.version sweval___{wildcards.dataset_combo}___{wildcards.test_dataset} \
            --wandb ''
    """

rule aggregate_spotwhisperer_test:
    """
    Aggregate retrieval and zero-shot metrics across test datasets for a model.
    """
    input:
        retrieval_results=lambda wildcards: expand(
            rules.spotwhisperer_test.output.retrieval_and_cwevals,
            dataset_combo=wildcards.dataset_combo,
            test_dataset=BASE_DATASETS,
        )
    output:
        aggregated_retrieval=BENCHMARKS_DIR / "retrieval" / "{dataset_combo}" / "aggregated_retrieval.csv",
        aggregated_cwevals=BENCHMARKS_DIR / "retrieval" / "{dataset_combo}" / "aggregated_cwevals.csv"
    conda:
        "cellwhisperer"
    params:
        dataset_combo=lambda wildcards: wildcards.dataset_combo.replace("__", ","),
        test_datasets=BASE_DATASETS
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/aggregate_retrieval_results.py"
