# matching concise and extensive questions
QUESTION_MAP = {
    "What does the sample represent?": "Provide an extensive description of the sample",
    "What does the transcriptome represent?": "What does the transcriptome represent? Respond comprehensively.",
    "What do these cells represent?": "Provide a detailed description of these cells.",
    "Give a brief description of the sample.": "Provide a detailed description of the sample.",
    "Give a brief description of these cells.": "Provide a detailed description of these cells.",
    "Present a compact description of the sample's key features.": "Present a detailed description of the sample's key features.",
    "Summarize the state of these cells.": "Provide a detailed description of the state of these cells.",
    "Provide a brief description of these cells.": "Provide a detailed description of these cells.",
    "Provide a brief description of the given sample.": "Provide a detailed description of the given sample.",
    "Describe the sample concisely.": "Describe the sample in detail.",
    "Describe these cells concisely.": "Describe these cells in detail.",
}

QUESTIONS = list(QUESTION_MAP.keys())

# random sample (n=100) of the GSVA datasets (which are already weight-corrected)
# Note: Weight correction for archs4 only used the transcriptome- not the annotation-embeddings. Due to this however, there are some single cells in here, which could count as data leakage. We used single_cell_probability > 0.2 and eliminated these samples in TEST_IDS and didn't observe any changes in the final scores.
TEST_IDS = pd.read_csv(PROJECT_DIR / "src/llava/test_ids", header=None).iloc[:, 0].to_list()

rule llava_stage1_dataset:
    """
    Generation of dataset for stage 1 training of LLaVA

    In a nutshell, take the questions (above) and use the previously generated sample annotations as answers
    """
    input:
        annotations_archs4_geo=ancient(PROJECT_DIR / config["paths"]["processed_multi_annotations"].format(dataset="archs4_geo")),
        annotations_cellxgene_census=PROJECT_DIR / config["paths"]["processed_annotations"].format(dataset="cellxgene_census"),
    output:
        train_set=PROJECT_DIR / config["paths"]["llava"]["pretrain_text_dataset"],
        test_set=PROJECT_DIR / config["paths"]["llava"]["root"] / "pretrain_test_set.json"
    params:
        seed=42,
        annotation_replicate=1,
        questions=QUESTIONS,
        transcriptome_tag="<image>",  # we stick to <image> because of the llava code base
        anndata_label_name=config["anndata_label_name"],
        test_ids=TEST_IDS
    conda:
        "cellwhisperer"
    notebook:
        "../notebooks/llava_stage1_dataset.py.ipynb"


rule prepare_llava_stage2_requests:
    """
    Generate metadata-based requests for the given inputs:

    - Processed annotation
    - GSVA
    - Top-most expressed genes

    Note: this generates requests for ALL 100k samples (include the ones for detailed and complex. this is necessary as well, for evaluation_generation)

    Few shot input format: JSON (top_gene_sets: list(ranked, avoid scores or ), top_genes: list(ranked), annotation:string, sample_id: string)
    """
    input:
        processed_annotations=PROJECT_DIR / config["paths"]["processed_multi_annotations"],
        gsva=PROJECT_DIR / config["paths"]["gsva"]["result"],
        top_genes=rules.compute_top_genes.output.top_genes,
        request_template=ancient("prompts/llava_stage2_request_template.txt"),
        few_shot_prompts=ancient(expand("prompts/llava_stage2_few_shot_samples/{i}_request.json", i=range(5))),
        few_shot_responses=ancient(expand("prompts/llava_stage2_few_shot_samples/{i}_response.json", i=range(5))),
    output:
        request_splits=scatter.split(PROJECT_DIR / "results/llava/requests/{{dataset,[^/]+}}/{scatteritem}.json"),
        few_shot_block="prompts/few_shot_messages_{dataset}.json",
    params:
        annotation_replicate=-1,  # -1 means the last one (which is the more sophisticated one for archs4_geo)
        top_n_genes=20,  # use fewer here to not overwhelm the model
        top_n_gene_sets=20,
        start_from_num=CONVERSATION_START,
    conda:
        "cellwhisperer"
    resources:
        mem_mb=400000,
        slurm="cpus-per-task=2"
    notebook:
        "../notebooks/prepare_llava_stage2_requests.py.ipynb"

rule generate_llava_stage2_conversations:
    """
    Generation of dataset for stage 2 training of LLaVA

    This is more sophisticated in that we use Mixtral (or GPT-4) to generate conversations from few-shot examples

    """
    input:
        model=PROJECT_DIR / config["model_name_path_map"]["mixtral"],
        instruction="prompts/llava_stage2_few_shot.txt",
        json_split=PROJECT_DIR / "results/llava/requests/{dataset}/{scatteritem}.json",
        json_schema="prompts/llava_stage2_schema.json",
        few_shot_messages="prompts/few_shot_messages_{dataset}.json", # rules.prepare_llava_stage2_requests.output.few_shot_block
    output:
        generated_conversations = protected(PROJECT_DIR / "results" / "llava" / "processed" / "{dataset,[^/]+}" / "{scatteritem}.json"),  # I marked this as protected as it might be costly to produce
    params:
        temperature=0.5,
        prompt_reminder= "\nMake sure to ignore any patient- or donor-specific information, like in your last responses."
    resources:
        mem_mb=100000,
        slurm=slurm_gres(num_cpus=25)
    conda: "llama_cpp"
    notebook: "../notebooks/llava_stage2_dataset.py.ipynb"


rule generate_llava_complex:
    """
    Stage 2 complex data generation (GPT-4)

    Currently *without* gene names, forcing the model to focus on the provided transcriptome.
    """
    input:
        processed_annotations=PROJECT_DIR / config["paths"]["processed_multi_annotations"],
        gsva=PROJECT_DIR / config["paths"]["gsva"]["result"],
        top_genes=ancient(rules.compute_top_genes.output.top_genes),
        system_message=ancient("prompts/llava_stage2_complex_few_shot.txt"),
        request_template=ancient("prompts/llava_stage2_request_template.txt"),
        few_shot_prompts=ancient(expand("prompts/llava_stage2_complex_few_shot_samples/{i}_request.json", i=[2, 3, 5])),
        few_shot_responses=ancient(expand("prompts/llava_stage2_complex_few_shot_samples/{i}_response.json", i=[2, 3, 5])),
    output:
        processed_annotation = protected(PROJECT_DIR / "results" / "llava" / "processed_complex" / "{dataset}" / "{sample_id}.json"),
    log:
        "logs/generate_llava_complex_{dataset}_{sample_id}.log"
    params:
        top_n_genes=50,
        top_n_gene_sets=50,
        annotation_replicate=1,
        temperature=0.4
    conda:
        # PROJECT_DIR / "envs" / "main.yaml"
        "cellwhisperer"
    resources:
        mem_mb=50000,
    script:
        "../scripts/generate_llava_complex.py"

rule generate_llava_detailed:
    """
    Stage 2 detailed data generation (GPT-4)

    # NOTE: some of the generated conversations failed (usually JSON generation errors)

    Gene names are only provided in the instructions (not in the user-part of the conversation), forcing the model to focus on the provided transcriptome embedding.
    """
    input:
        processed_annotations=PROJECT_DIR / config["paths"]["processed_multi_annotations"],
        gsva=PROJECT_DIR / config["paths"]["gsva"]["result"],
        top_genes=ancient(rules.compute_top_genes.output.top_genes),
        system_message=ancient("prompts/llava_stage2_detailed.txt"),
        request_template=ancient("prompts/llava_stage2_request_template.txt"),
    output:
        processed_annotation = protected(PROJECT_DIR / "results" / "llava" / "processed_detailed" / "{dataset}" / "{sample_id}.json"),  # I marked this as protected as it might be costly to produce
    log:
        "logs/generate_llava_detailed_{dataset}_{sample_id}.log"
    params:
        top_n_genes=50,
        top_n_gene_sets=50,
        annotation_replicate=1,
        temperature=0.0,
        question_map=QUESTION_MAP,
    conda:
        # PROJECT_DIR / "envs" / "main.yaml"
        "cellwhisperer"
    resources:
        mem_mb=50000,
    script:
        "../scripts/generate_llava_detailed.py"


rule aggregate_llava_stage2_dataset:
    """
    Read in all the generated annotations and aggregate them into a single JSON file

    NOTE consider oversampling complex items (e.g. use them twice, as they are fewer)
    """
    input:
        # json_splits = glob.glob("/msc/home/mschae83/cellwhisperer/results/llava{llava_dataset,[^/]*}/processed/archs4_geo/second/*-of-128.json"),  # , i=[1, 103, 118, 20, 35, 48, 61, 7, 89, 1, 104, 120, 23, 36, 5, 62, 74, 9, 102, 107, 122, 33, 37, 50, 65, 79, 93])
        json_splits=[split.format(dataset=dataset)
                     for dataset in ["archs4_geo", "cellxgene_census"]
                     for split in gather.split(PROJECT_DIR / "results" / "llava" / "processed" / "{{dataset}}" / "{scatteritem}.json")
                     # for split in [(PROJECT_DIR / "results" / "llava" / "processed" / "{dataset}" / f"1-of-128.json").as_posix()]
                     ],
        stage1_train_set = rules.llava_stage1_dataset.output.train_set,
        stage1_test_set = rules.llava_stage1_dataset.output.test_set,
        complex_conversations=ancient(expand(rules.generate_llava_complex.output.processed_annotation, sample_id=COMPLEX_SAMPLES, dataset="archs4_geo")),
        detailed_conversations=ancient(expand(rules.generate_llava_detailed.output.processed_annotation, sample_id=DETAILED_SAMPLES, dataset="archs4_geo")),
        transcriptome_weights=expand(PROJECT_DIR / "results/pre_training_processing/{dataset}/transcriptome_weights.npz", dataset=["archs4_geo", "cellxgene_census"]),
        annotation_weights=expand(PROJECT_DIR / "results/pre_training_processing/{dataset}/annotation_weights.npz", dataset=["archs4_geo", "cellxgene_census"])
    params:
        test_ids=TEST_IDS,
        num_stage1_samples=10000,  # maybe only use 10000. or train with them only in the beginning?
        seed=42,
        transcriptome_tag="<image>",  # we stick to <image> because of the llava code base
        accept_num_erroneous_jsons=200,
    log:
        "logs/aggregate_llava_stage2_dataset.log"
    resources:
        mem_mb=100000,
    conda:
        # PROJECT_DIR / "envs" / "main.yaml"
        "cellwhisperer"
    output:
        llava_stage2_dataset=PROJECT_DIR / config["paths"]["llava"]["finetune_text_dataset"].format(llava_dataset="_default"),
        evaluation_dataset=PROJECT_DIR / config["paths"]["llava"]["evaluation_text_dataset"].format(dataset="main_uncurated", llava_dataset="_default")
    script:
        "../scripts/aggregate_llava_stage2_dataset.py"

rule curate_evaluation_dataset:
    input:
        rules.aggregate_llava_stage2_dataset.output.evaluation_dataset,
    output:
        evaluation_dataset=PROJECT_DIR / config["paths"]["llava"]["evaluation_text_dataset"].format(dataset="main", llava_dataset="_default")
    shell: """
        echo "Please copy the evaluation dataset to the correct location and curate it manually. /home/moritz/Projects/cellwhisperer/src/experiments/422_curate_llava_testset/README.md"
        cp /home/moritz/Projects/cellwhisperer/src/experiments/422_curate_llava_testset/curated.json {output.evaluation_dataset}
    """


rule generate_cellxgene_census_conversations:
    """
    Generate a conversation training dataset based on cellxgene_census.
    Each conversation contains the question "what is the type of this cell?" and the response "this is a <celltype>".
    """
    input:
        annotations_cellxgene_census=PROJECT_DIR / config["paths"]["read_count_table"].format(dataset="cellxgene_census"),
        top_genes=PROJECT_DIR / "results" / "cellxgene_census" / "top_genes.parquet"
    output:
        PROJECT_DIR / config["paths"]["llava"]["finetune_text_dataset"]
    wildcard_constraints:
        llava_dataset="_celltype|_top50genescelltype}"
    params:
        question=config["llava_eval"]["question_celltype"],
        response_prefix=config["llava_eval"]["response_prefix_celltype"],
        pre_prompt_topgenes=lambda wildcards: config["llava_eval"]["pre_prompt_topgenes"] if "top50genes" in wildcards.llava_dataset else None,
        top_n_genes=50,
    conda:
        "cellwhisperer"
    resources:
        mem_mb=200000,
        # slurm=slurm_gres(num_gpus=5, num_cpus=40)
        slurm=slurm_gres("large", num_gpus=1, num_cpus=10)
    script:
        "../scripts/generate_cellxgene_census_conversations.py"
