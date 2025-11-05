rule geo_umap_plots:
    """
    TODO would still need to be updated for UCE-style model
    """
    input:
        adata=HTTP.remote(f"{config['precomputing_base_url']}/datasets/archs4_geo/cellxgene.h5ad", keep_local=True)[0],
        # adata=PROJECT_DIR / "results" / "archs4_geo" / "{model}" / "cellxgene.h5ad"  # The pipeline could compute it by itself, but it requires a lot of RAM and time
    output:
        cluster_labeled=PROJECT_DIR / config["paths"]["geo_umap"] / "cluster_labeled.svg",
        submission_date_labeled=PROJECT_DIR / config["paths"]["geo_umap"] / "submission_date_labeled.svg",
        highlighted_clusters_date_kdes=PROJECT_DIR / config["paths"]["geo_umap"] / "highlighted_clusters_date_kdes.svg",
    resources:
        mem_mb=300000,  # for good measure
        slurm="cpus-per-task=2"
    params:
        highlight_clusters=["CD34+ HSPCs with broad differentiation potential",
        "K562 leukemia cells cultured in supplemented RPMI 1640",
        "Active remodeling and immune response in cells"]
    conda:
        "cellwhisperer"
    log:
        notebook="../log/geo_umap_plots_{model}.log.ipynb"
    notebook:
        "../notebooks/geo_umap_plots.py.ipynb"


rule final_model_retrieval_scores:
    """
    Run "cellwhisperer test" on the final model and get the retrieval scores

    This just plots the retrieval scores (on the disease test set). To compute these values run.

    `cellwhisperer test --config src/experiments/408_ablation_study3/base_config.yaml --model_ckpt results/models/jointemb/cellwhisperer_clip_v1.ckpt --seed_everything 0`

    """
    input:
        csv=HTTP.remote(f"{config['precomputing_base_url']}/misc/human_disease_dedup_recall_at_5_wandb_export.csv", keep_local=False)[0],
        # model=PROJECT_DIR / config["paths"]["jointemb_models"] / f"{CLIP_MODEL}.ckpt",  # needed in theory to compute the retrieval scores
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        barplot=PROJECT_DIR / "results" / "plots" / "retrieval_scores" / "barplot.svg",
        lineplot=PROJECT_DIR / "results" / "plots" / "retrieval_scores" / "recall_at_5_lineplot.svg",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=20000,
        # slurm=slurm_gres()
    notebook:
        "../notebooks/final_model_retrieval_scores.py.ipynb"

rule plot_ablation_wandb:
    """
    The plotted scores here were originally exported from wandb. We provide them for your convenience now, but you can run the full ablations using the pipeline in `src/ablation_study`
    """
    input:
        csv=HTTP.remote(f"{config['precomputing_base_url']}/misc/ablation_study_wandb_export.csv", keep_local=True)[0]
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

rule fig1_main:
    input:
        rules.geo_umap_plots.output.cluster_labeled.format(model=CLIP_MODEL),
        # Figure S1
        rules.final_model_retrieval_scores.output.barplot,
        rules.final_model_retrieval_scores.output.lineplot,
        rules.plot_ablation_wandb.output.all_models_comparison,  # Ablation study
