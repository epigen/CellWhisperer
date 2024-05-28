"""
comparison models
"""


rule gpt4transcriptome_baseline:
    """
    Run GPT-4 with gene- and gene-set information (top 50)
    """
    input:
        transcriptome_text_features=rules.llava_evaluation_generation_preparation.output.formatted_questions_text_only,
    output:
        protected(PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_gpt4transcriptome_responses.jsonl")
    conda:
        "llava2"
    notebook:
        "../notebooks/gpt4transcriptome_baseline.ipynb"

rule llava_eval_gpt4_review:
    """
    Cost to run this (for 200 samples): ~3 USD

    The rule file is derived, but heavily adjusted from `LLaVA/llava/eval/table/rule.json`.

    TODO are `formatted_questions` the real formatted questions? I thought they are only the contexts DEBUG

    NOTE: don't run this on the tabula_sapiens evaluation dataset, as that one just contains the cell type names. (would require a different prompt)
    """
    input:
        questions=rules.llava_evaluation_generation_preparation.output.formatted_questions,
        reference_responses=rules.llava_evaluation_generation_preparation.output.reference_responses,
        llava_responses=expand(rules.llava_evaluation_generation.output.llava_responses, input_features=["", "_with_top_genes", "_with_top_genes_gene_sets", "_text_only"], allow_missing=True),
        gpt4transcriptome_baseline_responses=rules.gpt4transcriptome_baseline.output,
        rule=ancient("prompts/gpt_evaluation_prompts.json")
    output:
        evaluation=protected(PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "generation_gpt4_review.jsonl")
    conda:
        "llava2"
    params:
        script=PROJECT_DIR / "modules/LLaVA/llava/eval/eval_gpt_review.py",
    shell: """
    python {params.script} -q {input.questions} -a {input.llava_responses} {input.gpt4transcriptome_baseline_responses} {input.reference_responses} -r {input.rule} -o {output[0]}
    """

rule llava_eval_gpt4_review_summarize:
    """
    # - compare (group by normal, complex, detailed, cellxgene/archs4)
    """

    input:
        evaluation=rules.llava_eval_gpt4_review.output.evaluation,
        archs4_data=PROJECT_DIR / config["paths"]["read_count_table"].format(dataset="archs4_geo"),  # provides the option to exclude single cells
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
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
