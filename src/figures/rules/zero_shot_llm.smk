rule zero_shot_llm_prediction:
    """
    GPT-4 zero-shot predictions, adapting prompt from Hou et al., 2024 (https://www.nature.com/articles/s41592-024-02235-4)

    For Llama 3.3 and DeepSeek, run first `export OLLAMA_HOST=0.0.0.0:8080 ~/ollama/bin/ollama serve` (note that they might not operate well simulaneously)

    """
    input:
        read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        gene_normalizers=rules.compute_gene_normalizers.output.gene_mean_log1ps,  # NOTE could also get the ones computed with `seurat_get_top_genes`
    output:
        predictions=protected(ZERO_SHOT_RESULTS / "{model,gpt4|llama33|claudesonnet|deepseek|llama31|mistral7b}" / "datasets" / "{dataset,[^/]+}" / "predictions" / "{metadata_col}.{grouping,by_cell|by_class}.csv"),
    params:
        api_key=lambda wildcards: os.getenv(config["zero_shot_llms"][wildcards.model]["api_key_env"]),
        api_base_url=lambda wildcards: config["zero_shot_llms"][wildcards.model]["base_url"],
        prompt=lambda wildcards: f"Identify the {wildcards.metadata_col} for a given set of markers. Only provide the name of the {wildcards.metadata_col}. Do not show numbers before the name. \n{wildcards.metadata_col} candidates: {{candidates}}\n\nMarkers: {{markers}}",
        top_n_genes=50,
        model=lambda wildcards: config["zero_shot_llms"][wildcards.model]["model_name"],
        average_by_class=lambda wildcards: wildcards.grouping == "by_class",
    resources:
        mem_mb=120000,  # TODO check if really enough..
        slurm="cpus-per-task=2"
    conda:
        "cellwhisperer"
    log:
        notebook="../logs/zero_shot_llm_prediction_{model}_{dataset}_{metadata_col}_{grouping}.ipynb",
        progress="../logs/zero_shot_llm_prediction_{model}_{dataset}_{metadata_col}_{grouping}.log"
    notebook:
        "../notebooks/zero_shot_llm_prediction.py.ipynb"


# dataset_selector=lambda wildcards:
def dataset_selector(wildcards):
    return [dataset for dataset, mcols in config["metadata_cols_per_zero_shot_validation_dataset"].items()
                                    if wildcards.metadata_col in mcols
                                    and (  # exclude some of the tabula_sapiens datasets (to save costs on the API)
                                        (wildcards.grouping == "by_cell" and dataset not in ["tabula_sapiens", "tabula_sapiens_well_studied_celltypes"]) or
                                        (wildcards.grouping == "by_class" and dataset != "tabula_sapiens_100_cells_per_type")
                                    )]
rule aggregate_zero_shot_llm_property_predictions:
    """
    Aggregate the zero-shot property predictions for all models and datasets

    grouping: indicate whether the feature matrix is grouped and mean-aggregated prior to prediction. Either "by_class" or "by_cell"

    TODO organ_tissue not working
    """
    input:
        predictions=lambda wildcards: [
            ZERO_SHOT_RESULTS / model / "datasets" / dataset / "predictions" / f"{wildcards.metadata_col}.{wildcards.grouping}.csv"
            for model in ZERO_SHOT_PREDICTORS
            for dataset in dataset_selector(wildcards)
        ]
    output:
        aggregated_predictions=ZERO_SHOT_RESULTS / "aggregated_predictions_{metadata_col}_{grouping,by_cell|by_class}.csv",
        aggregated_predictions_plot=ZERO_SHOT_RESULTS / "aggregated_predictions_{metadata_col}_{grouping,by_cell|by_class}.png"
    params:
        metric="accuracy",
        models=ZERO_SHOT_PREDICTORS,
        datasets=dataset_selector,
        plot_title=lambda wildcards: f"accuracy for {wildcards.metadata_col} ({wildcards.grouping})"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=2000,
        slurm="cpus-per-task=1"
    log:
        notebook="../logs/aggregate_zero_shot_llm_property_predictions_{metadata_col}_{grouping}.ipynb"
    notebook:
        "../notebooks/aggregate_zero_shot_llm_property_predictions.py.ipynb"
