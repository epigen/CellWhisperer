rule zero_shot_llm_prediction_download:
    """
    For reproducibility and to avoid OpenAI API calls, we provide precomputed results
    """
    input:
        HTTP.remote(f"{config['precomputing_base_url']}/datasets/{{dataset}}/zero_shot_llm/{{model}}_{{metadata_col}}_{{grouping}}.csv", keep_local=False)[0],
    output:
        predictions=ZERO_SHOT_RESULTS / "{model,gpt4|llama33|claudesonnet|deepseek|mistral7b}" / "datasets" / "{dataset,[^/]+}" / "predictions" / "{metadata_col}.{grouping,by_cell|by_class}.csv",
    run:
        import shutil
        shutil.copy(input[0], output.predictions)

# rule zero_shot_llm_prediction:
#     """
#     GPT-4 zero-shot predictions, adapting prompt from Hou et al., 2024 (https://www.nature.com/articles/s41592-024-02235-4)

#     For Llama 3.1 and 3.3 and Mistral, run first `export OLLAMA_HOST=0.0.0.0:8080 ~/ollama/bin/ollama serve` (note that they might not operate well simultaneously)

#     """
#     input:
#         read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
#         gene_normalizers=rules.compute_gene_normalizers.output.gene_mean_log1ps,  # NOTE could also get the ones computed with `seurat_get_top_genes`
#     output:
#         predictions=protected(ZERO_SHOT_RESULTS / "{model,gpt4|llama33|claudesonnet|deepseek|mistral7b}" / "datasets" / "{dataset,[^/]+}" / "predictions" / "{metadata_col}.{grouping,by_cell|by_class}.csv"),
#     params:
#         api_key=lambda wildcards: os.getenv(config["llm_apis"][wildcards.model]["api_key_env"]),
#         api_base_url=lambda wildcards: config["llm_apis"][wildcards.model]["base_url"],
#         prompt=lambda wildcards: f"Identify the {wildcards.metadata_col} for a given set of markers. Only provide the name of the {wildcards.metadata_col}. Do not show numbers before the name. \n{wildcards.metadata_col} candidates: {{candidates}}\n\nMarkers: {{markers}}",
#         top_n_genes=50,
#         model=lambda wildcards: config["llm_apis"][wildcards.model]["model_name"],
#         average_by_class=lambda wildcards: wildcards.grouping == "by_class",
#     resources:
#         mem_mb=lambda wildcards: 300000 if wildcards.dataset == "tabula_sapiens" else 120000,
#         slurm="cpus-per-task=2"
#     conda:
#         "cellwhisperer"
#     log:
#         notebook="../logs/zero_shot_llm_prediction_{model}_{dataset}_{metadata_col}_{grouping}.ipynb",
#         progress="../logs/zero_shot_llm_prediction_{model}_{dataset}_{metadata_col}_{grouping}.log"
#     notebook:
#         "../notebooks/zero_shot_llm_prediction.py.ipynb"

rule integrate_zero_shot_embedding_performance:
    input:
        predictions_raw=rules.zero_shot_cellwhisperer_prediction.output.predictions.replace("{grouping}", "by_cell"),
    output:
        predicted_labels=ZERO_SHOT_CW_MODEL_RESULTS / "datasets" / "{dataset,[^/]+}" / "predicted_labels" / "{metadata_col}.{grouping,by_cell|by_class}.csv",  # NOTE: might be used as input for `plot_confusion_matrix` as well (and by extension `zero_shot_performance_macroavg`)
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=2"
    conda:
        "cellwhisperer"
    notebook:
        "../notebooks/integrate_zero_shot_embedding_performance.py.ipynb"


# llava_dataset, base_model, model, prompt_variation
LLM_PPL_PREDS = [
    ("_default", config['model_name_path_map']['llava_base_llm'], "NONE", "with50topgenesresponsenoembedding"),  # Vanilla Mistral  # `_default` works and is good for comparability to cellwhisperer_clip_v1
    ("_top50genescelltype", config['model_name_path_map']['llava_base_llm'],  "uce", "with50topgenesresponsenoembedding"),  # this one was trained with "uce" in the path name, but it does not use the embedding in any case..
    ("_celltype", config['model_name_path_map']['llava_base_llm'], "cellwhisperer_clip_v1", "without50topgenesresponse"),
    ("_default", config['model_name_path_map']['llava_base_llm'], "cellwhisperer_clip_v1", "without50topgenesresponse"),

]


rule aggregate_zero_shot_llm_property_predictions:
    """
    Only works for cell types

    Aggregate the zero-shot property predictions for all models and datasets

    grouping: indicate whether the feature matrix is grouped and mean-aggregated prior to prediction. Either "by_class" or "by_cell"

    # pancreas does not work, presumably because its cell types are not very common. Same problem as the previous Ext. Data Fig 3b
    NOTE: f1_score is macro-averaging
    """
    input:
        predictions=lambda wildcards: [
            ZERO_SHOT_RESULTS / model / "datasets" / "tabula_sapiens_100_cells_per_type" / "predictions" / f"{wildcards.metadata_col}.{wildcards.grouping}.csv"
            for model in config["llm_apis"].keys()
        ] + [
            rules.llava_evaluation_prediction_scores.output.predictions.format(
                llava_dataset=llava_dataset,
                dataset="tabula_sapiens_100_cells_per_type",
                base_model=base_model,
                model=model,
                prompt_variation=prompt_variation)
            for llava_dataset, base_model, model, prompt_variation in LLM_PPL_PREDS
        ] + [
            # rules.integrate_zero_shot_embedding_performance.output.predicted_labels.format(
            rules.zero_shot_cellwhisperer_prediction.output.predictions.replace("{grouping}", "by_cell").format(
                dataset="tabula_sapiens_100_cells_per_type",
                metadata_col="celltype",
                model=model,
                grouping="by_cell")
            for model in ["cellwhisperer_clip_v1", "cellwhisperer_clip_v2_uce", "cellwhisperer_clip_v2_scgpt"]
        ]
    output:
        aggregated_predictions=ZERO_SHOT_RESULTS / "aggregated_predictions_{metadata_col}_{grouping,by_cell|by_class}.csv",
        aggregated_predictions_plot=ZERO_SHOT_RESULTS / "aggregated_predictions_{metadata_col}_{grouping,by_cell|by_class}.svg"
    params:
        metric="accuracy",   # "f1_score" also works
        models=list(config["llm_apis"].keys()) + ["__".join(v) for v in LLM_PPL_PREDS] + ["cellwhisperer_clip_v1", "cellwhisperer_clip_v2_uce", "cellwhisperer_clip_v2_scgpt"],
        datasets=["tabula_sapiens_100_cells_per_type"],
        plot_title=lambda wildcards: f"accuracy for {wildcards.metadata_col} ({wildcards.grouping})",
        costs=[  # computed as in `cellwhisperer/src/experiments/702-cw_benefit_llm/README.md`
            937.5,  # GPT-4o
            750.0,  # Claude Sonnet
            667.0,  # Llama 3.3 70B
            66.7,  # Mistral 7B
            13.3,  # Mistral 7B*
            13.3,  # Mistral 7B* (cell type-focused)
            6.0,  # CellWhisperer* (cell type-focused)
            6.0,  # CellWhisperer*
            0.6,  # CW embedding
            3.3,  # CW embedding (UCE)
            0.33,  # CW embedding (scGPT)
        ],
        labels=[
            "GPT-4o",
            "Claude Sonnet 3.5",
            "Llama 3.3 70B",
            "Mistral 7B",
            "Mistral 7B",
            "Mistral 7B (cell type fine-tuned)",
            "CW chat (cell type fine-tuned)",
            "CW chat",
            "CW embed.",
            "CW embed. (UCE)",
            "CW embed. (scGPT)",
        ]
    conda:
        "cellwhisperer"
    resources:
        mem_mb=2000,
        slurm="cpus-per-task=1"
    log:
        notebook="../logs/aggregate_zero_shot_llm_property_predictions_{metadata_col}_{grouping}.ipynb"
    notebook:
        "../notebooks/aggregate_zero_shot_llm_property_predictions.py.ipynb"
