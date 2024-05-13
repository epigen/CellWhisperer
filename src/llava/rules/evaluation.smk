"""
still relatively limited. next better evaluations would include

- how many of the top 100 genes can it recover?
- how well does it reproduce pathways?
"""


rule llava_evaluation_perplexity:
    """
    Current limitations:
    - Each of our datasets can only provide one evaluation dataset
    """
    input:
        llava_model=ancient(rules.finetune_llava.output.output_dir),
        evaluation_dataset=PROJECT_DIR / config["paths"]["llava"]["evaluation_text_dataset"],
        # image_data=rules.process_full_dataset.output.model_outputs.format(dataset="{dataset}", model=config["model_name_path_map"]["cellwhisperer"]),
        image_data=rules.combine_processed_data.output.combined.format(model=config["model_name_path_map"]["cellwhisperer"]),
    conda:
        "llava2"
    output:
        all_perplexities=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "all_perplexities.csv",
    params:
        num_projector_tokens=int(PROJECTOR_TYPE.split("_")[1].strip("t")),
        background_shuffle=lambda wildcards: "transcriptome" if wildcards.dataset == "archs4_metasra" else "llm-response",
        num_negatives=30
    resources:
        mem_mb=300000,
        slurm=f"cpus-per-task=40 gres=gpu:{GPU_TYPE}:1 qos={GPU_TYPE} partition=gpu"
    log:
        "logs/llava_evaluation_perplexity/{dataset}_{base_model}_{model}.log"
    threads: 16
    notebook:
        "../notebooks/llava_evaluation_perplexity.py.ipynb"


rule llava_evaluation_perplexity_plots:
    """
    # TODO need to aggregate like this (more or less)

    """

    input:
        all_perplexities=rules.llava_evaluation_perplexity.output.all_perplexities,
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        log_perplexity_ratio=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "log_mean_perplexity.ratio",  # smaller is better log(ppl_real/ppl_neg_control)
        comparison_plot=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "perplexity_quantile.svg",  # barplot
        detailed_plot=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "detailed.svg",  # barplot
    conda:
        "cellwhisperer"
    notebook:
        "../notebooks/llava_evaluation_perplexity_plots.py.ipynb"

rule llava_evaluation_generation_preparation:
    """
    Extract the first question for each conversation (if there are more than one questions) and bring it into the right format for the evaluation
    """

    input:
        evaluation_dataset=PROJECT_DIR / config["paths"]["llava"]["evaluation_text_dataset"],
        archs4_request_splits=gather.split(PROJECT_DIR / "results/llava/requests/archs4_metasra/{scatteritem}.json"),
        cellxgene_request_splits=gather.split(PROJECT_DIR / "results/llava/requests/cellxgene_census/{scatteritem}.json"),
    output:
        formatted_questions=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_formatted_questions.jsonl",
        formatted_questions_with_top_genes=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_formatted_questions_with_top_genes.jsonl",
        formatted_questions_with_top_genes_gene_sets=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_formatted_questions_with_top_genes_gene_sets.jsonl",
        formatted_questions_text_only=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_formatted_questions_text_only.jsonl",
        reference_responses=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_reference_responses.jsonl",
    params:
        instruction_prompt=INSTRUCTION_PROMPT,
        instruction_response=lambda wildcards: INSTRUCTION_RESPONSE,
        instruction_response_gene_sets_extension=lambda wildcards: INSTRUCTION_RESPONSE_GENE_SETS_EXTENSION,
        instruction_prompt_text_only=lambda wildcards: INSTRUCTION_PROMPT_TEXT_ONLY,
    conda:
        "llava2"
    notebook:
        "../notebooks/llava_evaluation_generation_preparation.py.ipynb"


rule llava_evaluation_generation:
    """
    Generate responses for the test questions for later evaluation
    """

    input:
        llava_model=ancient(rules.finetune_llava.output.output_dir),
        image_data=rules.combine_processed_data.output.combined.format(model=config["model_name_path_map"]["cellwhisperer"]),
        questions=lambda wildcards: rules.llava_evaluation_generation_preparation.output[f"formatted_questions{wildcards.input_features}"],
    output:
        llava_responses=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_llava_responses{input_features,[^/]*}.jsonl"
    conda:
        "llava2"
    params:
        script=lambda wildcards: PROJECT_DIR / "modules/LLaVA/llava/eval" / ("model_qa.py" if wildcards.input_features == "_text_only" else "model_vqa.py"),
    resources:
        mem_mb=100000,
        slurm=f"cpus-per-task=40 gres=gpu:{GPU_TYPE}:1 qos={GPU_TYPE} partition=gpu"
    shell: """
    if [ "{wildcards.input_features}" == "_text_only" ]; then
        python {params.script} --model-name mistralai/Mistral-7B-Instruct-v0.2 --question-file {input.questions} --answers-file {output.llava_responses}
    else
        python {params.script} --num_beams 10 --conv-mode mistral_instruct --temperature 0.0 --model-path {input.llava_model} --image-data {input.image_data} --question-file {input.questions} --answers-file {output.llava_responses}
    fi
    """
