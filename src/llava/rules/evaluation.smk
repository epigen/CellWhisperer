"""
still relatively limited. next better evaluations would include

- how many of the top 100 genes can it recover?
- how well does it reproduce pathways?
"""

rule tabsap_celltype_evaluation_dataset:
    input:
        dataset=PROJECT_DIR / config["paths"]["read_count_table"].format(dataset="tabula_sapiens"),
    output:
        evaluation_dataset=PROJECT_DIR / config["paths"]["llava"]["evaluation_text_dataset"].format(dataset="tabula_sapiens"),
    params:
        celltypes=config["top20_lung_liver_blood_celltypes"],  # TODO use all?
        num_cells_per_celltype=20,
        question="Which cell type is this cell?"
    conda:
        "cellwhisperer"
    notebook:
        "../notebooks/tabsap_celltype_evaluation_dataset.py.ipynb"


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
        log_perplexity_ratio=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "log_mean_perplexity.ratio",  # smaller is better log(ppl_real/ppl_neg_control)
        all_perplexities=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "all_perplexities.csv",
        comparison_plot=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "comparison.svg"
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
        reference_responses=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_reference_responses.jsonl",
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
        questions=rules.llava_evaluation_generation_preparation.output.formatted_questions,
    output:
        llava_responses=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_llava_responses.jsonl"
    conda:
        "llava2"
    params:
        script=PROJECT_DIR / "modules/LLaVA/llava/eval/model_vqa.py",
    resources:
        mem_mb=100000,
        slurm=f"cpus-per-task=40 gres=gpu:{GPU_TYPE}:1 qos={GPU_TYPE} partition=gpu"
    shell: """
    python {params.script} --num_beams 10 --conv-mode mistral_instruct --temperature 0.0 --model-path {input.llava_model} --image-data {input.image_data} --question-file {input.questions} --answers-file {output.llava_responses}
    """

rule llava_eval_gpt4_review:
    """
    Cost to run this (for 200 samples): ~3 USD

    The rule file is derived, but heavily adjusted from `LLaVA/llava/eval/table/rule.json`.

    NOTE: don't run this on the tabula_sapiens evaluation dataset, as that one just contains the cell type names. (would require a different prompt)
    """
    input:
        questions=rules.llava_evaluation_generation_preparation.output.formatted_questions,
        reference_responses=rules.llava_evaluation_generation_preparation.output.reference_responses,
        llava_responses=rules.llava_evaluation_generation.output.llava_responses,
        rule="prompts/gpt_evaluation_prompts.json"
    output:
        evaluation=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_gpt4_review.jsonl"
    conda:
        "llava2"
    params:
        script=PROJECT_DIR / "modules/LLaVA/llava/eval/eval_gpt_review.py",
    shell: """
    python {params.script} -q {input.questions} -a {input.llava_responses} {input.reference_responses} -r {input.rule} -o {output[0]}
    """

rule llava_eval_gpt4_review_summarize:
    """
    # - compare (group by normal, complex, detailed, cellxgene/archs4)
    """

    input:
        evaluation=rules.llava_eval_gpt4_review.output.evaluation,
        archs4_data=PROJECT_DIR / config["paths"]["read_count_table"].format(dataset="archs4_metasra"),
    output:
        overview_plot=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_gpt4_review_summary.svg"
    conda:
        "llava2"
    params:
        complex_samples=COMPLEX_SAMPLES,  # for grouping
        detailed_samples=DETAILED_SAMPLES,  # for grouping
    log:
        "logs/llava_eval_gpt4_review_summarize_{dataset}_{base_model}__{model}.log"
    notebook:
        "../notebooks/llava_eval_gpt4_review_summarize.py.ipynb"
