# TODO move to config
API_KEYS = {
    "gpt4": os.getenv("OPENAI_API_KEY"),
    "llama33": "AAAAC3NzaC1lZDI1NTE5AAAAIHBNHW83n+Y6CxrSudipeoRn7qC0EjnVRh00pWHn/oNy",
    # TODO "claudesonnet": os.getenv("ANTHROPIC_API_KEY"),  # TODO opus is even better apparently
}

API_BASE_URLS = {
    "gpt4": "https://api.openai.com/v1/",
    "llama33": "http://s0-n02.hpc.meduniwien.ac.at:8080/v1/",
    # TODO "claudesonnet": "https://api.anthropic.com/v1/completions",
}

MODEL_NAMES = {
    "gpt4": "gpt-4o-2024-11-20",
    "llama33": "llama3.3:70b",
    # "claudesonnet": "claudesonnet",  TODO
    # "claudeopus": "claudeopus",  TODO
}

rule zero_shot_llm_prediction:
    """
    GPT-4 zero-shot predictions, adapting prompt from Hou et al., 2024 (https://www.nature.com/articles/s41592-024-02235-4)
    For Llama 3.3, run first `export OLLAMA_HOST=0.0.0.0:8080 ~/ollama/bin/ollama serve`

    NOTE 2: for now we do cell type level predictions. better would be for each cell, or at least for each pseudo-cell (how to aggregate?). Alternatively, I could predict (all) cells with LLaMA 3
    """
    input:
        read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        gene_normalizers=rules.compute_gene_normalizers.output.gene_mean_log1ps,  # NOTE could also get the ones computed with `seurat_get_top_genes`
    output:
        # Using a directory here because the exact files produced depend on the dataset:
        predictions=protected(ZERO_SHOT_RESULTS / "{model,gpt4|llama33|sonnet}" / "datasets" / "{dataset,[^/]+}" / "predictions" / "{metadata_col}.{grouping,by_cell|by_class}.csv"),
    params:
        api_key=lambda wildcards: API_KEYS.get(wildcards.model),
        api_base_url=lambda wildcards: API_BASE_URLS.get(wildcards.model),
        prompt=lambda wildcards: f"Identify the {wildcards.metadata_col} for a given set of markers. Only provide the name of the {wildcards.metadata_col}. Do not show numbers before the name. \n{wildcards.metadata_col} candidates: {{candidates}}\n\nMarkers: {{markers}}",
        top_n_genes=50,
        model=lambda wildcards: MODEL_NAMES.get(wildcards.model),
        average_by_class=lambda wildcards: wildcards.grouping == "by_class",
    resources:
        mem_mb=350000,
        slurm="cpus-per-task=2"
    conda:
        "cellwhisperer"
    log:
        notebook="../logs/zero_shot_llm_prediction_{model}_{dataset}_{metadata_col}_{grouping}.ipynb"
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
rule aggregate_zero_shot_property_predictions:
    """
    Aggregate the zero-shot property predictions for all models and datasets

    TODO implement

    grouping: indicate whether the feature matrix is grouped and mean-aggregated prior to prediction. Either "by_class" or "by_cell"
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
        datasets=dataset_selector
    conda:
        "cellwhisperer"
    resources:
        mem_mb=2000,
        slurm="cpus-per-task=1"
    log:
        notebook="../logs/aggregate_zero_shot_property_predictions_{metadata_col}_{grouping}.ipynb"
    notebook:
        "../notebooks/aggregate_zero_shot_property_predictions.py.ipynb"

rule seurat_get_top_genes:
    """
    TODO still unused

    """
    input:
        raw_read_count_table=PROJECT_DIR / config["paths"]["read_count_table"]
    conda:
        "../../../envs/gpt4_hou2024.yaml"
    resources:
        mem_mb=350000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/zero_shot_gpt4_prediction_{dataset}_{metadata_col}.ipynb"
    notebook:
        "../notebooks/zero_shot_gpt4_prediction.r.ipynb"