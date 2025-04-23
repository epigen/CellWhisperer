rule llava_evaluation_generation_preparation:
    """
    Extract the first question for each conversation (if there are more than one questions) and bring it into the right format for the evaluation
    """

    input:
        evaluation_dataset=PROJECT_DIR / config["paths"]["llava"]["evaluation_text_dataset"],
        full_data=expand(PROJECT_DIR / config["paths"]["full_dataset"], dataset=["archs4_geo", "cellxgene_census"]),  # for annotations
        top_genes=expand(rules.compute_top_genes.output.top_genes, dataset=["archs4_geo", "cellxgene_census"]),
    output:
        formatted_questions=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_formatted_questions.jsonl",
        formatted_questions_with_top_genes=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_formatted_questions_with_top_genes.jsonl",
        formatted_questions_text_only=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_formatted_questions_text_only.jsonl",
        reference_responses=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_reference_responses.jsonl",
    params:
        request_template=lambda wildcards: config["llm_judge"]["request_template"],
        instruction_prompt=config["llm_judge"]["instruction_prompt"],
        instruction_response=lambda wildcards: config["llm_judge"]["instruction_response"],
        instruction_prompt_text_only=lambda wildcards: config["llm_judge"]["instruction_prompt_text_only"],
        top_n_genes=50,
    conda:
        "llava"
    notebook:
        "../notebooks/llava_evaluation_generation_preparation.py.ipynb"


rule llava_evaluation_generation:
    """
    Generate responses for the test questions for later evaluation

    TODO provide preprompt as in wrapper!!
    """

    input:
        llava_model=ancient(PROJECT_DIR / config["paths"]["llava"]["finetuned_model_dir"]),
        image_data=rules.combine_processed_data.output.combined.format(model=config["model_name_path_map"]["cellwhisperer"]),
        questions=lambda wildcards: rules.llava_evaluation_generation_preparation.output[f"formatted_questions{wildcards.input_features}"],
    output:
        llava_responses=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_llava_responses{input_features,[^/]*}.jsonl"
    conda:
        "llava"
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

rule gpt4transcriptome_baseline:
    """
    Run GPT-4 with top-gene information (top 50)

    """
    input:
        transcriptome_text_features=rules.llava_evaluation_generation_preparation.output.formatted_questions_text_only,
    output:
        response=protected(PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_gpt4transcriptome_responses.jsonl")
    params:
        api_key=lambda wildcards: os.getenv(config["llm_apis"]["gpt4"]["api_key_env"]),
        api_base_url=lambda wildcards: config["llm_apis"]["gpt4"]["base_url"],
        model=lambda wildcards: config["llm_apis"]["gpt4"]["model_name"],
    conda:
        "llava"
    notebook:
        "../notebooks/gpt4transcriptome_baseline.py.ipynb"

rule llava_eval_gpt4_review:
    """
    Cost to run this (for 200 samples): ~3 USD

    The rule file is derived, but heavily adjusted, from `LLaVA/llava/eval/table/rule.json`.

    NOTE: don't run this on the tabula_sapiens evaluation dataset, as that one just contains the cell type names. (would require a different prompt)
    """
    input:
        questions=rules.llava_evaluation_generation_preparation.output.formatted_questions,  # Contains 'reference' data (top genes, gene sets, caption), the question ('text') and the transcriptome ID ('image')
        reference_responses=rules.llava_evaluation_generation_preparation.output.reference_responses,
        llava_responses=expand(rules.llava_evaluation_generation.output.llava_responses, input_features=["", "_text_only"], allow_missing=True),
        gpt4transcriptome_baseline_responses=rules.gpt4transcriptome_baseline.output.response,
        rule=ancient("../llava/prompts/gpt_evaluation_prompts.json")  # Alternative prompt (leading to similar results: ../llava/prompts/gpt_evaluation_prompts_alternative.json)
    output:
        evaluation=protected(PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_gpt4_review.jsonl")
    conda:
        "llava"
    params:
        script=PROJECT_DIR / "modules/LLaVA/llava/eval/eval_gpt_review.py",  # TODO: only supports gpt4 so far
        api_key=lambda wildcards: os.getenv(config["llm_apis"]["gpt4"]["api_key_env"]),
        api_base_url=lambda wildcards: config["llm_apis"]["gpt4"]["base_url"],
        model=lambda wildcards: config["llm_apis"]["gpt4"]["model_name"],
    shell: """
    API_KEY={params.api_key} python {params.script} -q {input.questions} -a {input.llava_responses} {input.gpt4transcriptome_baseline_responses} {input.reference_responses} -r {input.rule} -o {output[0]} --model {params.model}
    """
    #  --api-base-url {params.api_base_url}

rule llava_eval_gpt4_review_summarize:
    """
    # - compare (group by normal, complex, detailed, cellxgene/archs4)
    TODO drop the complex vs detailed comparison
    """

    input:
        evaluation=rules.llava_eval_gpt4_review.output.evaluation,
        archs4_data=PROJECT_DIR / config["paths"]["read_count_table"].format(dataset="archs4_geo"),  # provides the option to exclude single cells.
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        overview_plot=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_gpt4_review_summary.svg",
    conda:
        "llava"
    params:
        complex_samples=COMPLEX_SAMPLES,  # for grouping
        detailed_samples=DETAILED_SAMPLES,  # for grouping
    log:
        "logs/llava_eval_gpt4_review_summarize_{dataset}_{base_model}__{model}_{prompt_variation}.log"
    notebook:
        "../notebooks/llava_eval_gpt4_review_summarize.py.ipynb"

rule llava_eval_table:
    """
    Take all the jsonl output files and add them to an excel table
    """
    input:
        questions=rules.llava_evaluation_generation_preparation.output.formatted_questions,  # Contains 'reference' data (top genes, gene sets, caption), the question ('text') and the transcriptome ID ('image')
        reference_responses=rules.llava_evaluation_generation_preparation.output.reference_responses,
        llava_responses=expand(rules.llava_evaluation_generation.output.llava_responses, input_features=["", "_text_only"], allow_missing=True),    # TODO careful the notebook indexes them with [0], [1]
        gpt4transcriptome_baseline_responses=rules.gpt4transcriptome_baseline.output.response,
        llava_eval_gpt4_review=rules.llava_eval_gpt4_review.output.evaluation,
    output:
        llava_eval_table=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_llava_eval_table.xlsx",
    conda:
        "cellwhisperer"
    notebook:
        "../notebooks/llava_eval_table.py.ipynb"



rule llm_judge_all:
    input:
        rules.llava_eval_gpt4_review_summarize.output.overview_plot.format(base_model=LLAVA_BASE_MODEL, model=CLIP_MODEL, prompt_variation="llm_judge", dataset="main"),
        rules.llava_eval_table.output.llava_eval_table.format(base_model=LLAVA_BASE_MODEL, model=CLIP_MODEL, prompt_variation="llm_judge", dataset="main"),
