
rule final_model_retrieval_scores:
    """
    Run "cellwhisperer test" on the final model and get the retrieval scores

    This just plots the retrieval scores (on the disease test set). To compute these values run.

    `cellwhisperer test --config src/experiments/408_ablation_study3/base_config.yaml --model_ckpt results/models/jointemb/cellwhisperer_clip_v1.ckpt --seed_everything 0`

    """
    input:
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / f"{CLIP_MODEL}.ckpt",  # needed in theory to compute the retrieval scores
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        plot=PROJECT_DIR / "results" / "plots" / "retrieval_scores" / "barplot.svg"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=20000,
        slurm="cpus-per-task=5 gres=gpu:a100:1 qos=a100 partition=gpu"
    notebook:
        "../notebooks/final_model_retrieval_scores.py.ipynb"
