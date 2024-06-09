rule final_model_retrieval_scores:
    """
    Run "cellwhisperer test" on the final model and get the retrieval scores

    This just plots the retrieval scores (on the disease test set). To compute these values run.

    `cellwhisperer test --config src/experiments/408_ablation_study3/base_config.yaml --model_ckpt results/models/jointemb/cellwhisperer_clip_v1.ckpt --seed_everything 0`

    """
    input:
        # model=PROJECT_DIR / config["paths"]["jointemb_models"] / f"{CLIP_MODEL}.ckpt",  # needed in theory to compute the retrieval scores
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        plot=PROJECT_DIR / "results" / "plots" / "retrieval_scores" / "barplot.svg"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=20000,
        # slurm="cpus-per-task=5 gres=gpu:a100:1 qos=a100 partition=gpu"
    notebook:
        "../notebooks/final_model_retrieval_scores.py.ipynb"


from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
HTTP = HTTPRemoteProvider()
rule download_wandb_export:
    input:
        HTTP.remote(f"{config['precomputing_base_url']}/ablation_study_wandb_export.csv", keep_local=False)[0]
    output:
        PROJECT_DIR / "results/ablation_study_wandb_export.csv"
    shell: """
        echo 'These scores were originally exported from wandb. We provide them for your convenience now, but you can run the full ablations using the pipeline in `src/ablation_study`'
        cp {input} {output}
    """

rule plot_wandb:
    """
    """
    input:
        csv=ancient(rules.download_wandb_export.output[0])  # NOTE: CSV was obtained from wandb. For convenience, it is committed as part of the source code right now
    output:
        top_models_metrics_details=PROJECT_DIR / config["paths"]["ablations"]["plots"] / "top_models_metrics_details.pdf",
        all_models_comparison=PROJECT_DIR / config["paths"]["ablations"]["plots"] / "all_models_comparison.pdf",
    params:
        metrics = [
            "valfn_integration_TabSap/avg_bio",
            "valfn_integration_TabSap/ASW_label__batch",  # Note: relatively narrow result range (0.83 - 0.86)
            "valfn_human_disease_dedup/recall_at_5_macroAvg",
            "valfn_zshot_TabSap_cell_lvl/f1_macroAvg",
            "valfn_zshot_TabSap_celltype_lvl/f1_macroAvg"],
        top_models = ["full_model", "no_archs4_data", "no_census_data", "scgpt"]
    conda:
        "cellwhisperer"
    notebook:
        "../notebooks/plot_wandb.py.ipynb"
