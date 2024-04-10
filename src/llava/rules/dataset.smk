rule compute_top_genes:
    """
    Compute the top genes for each sample based on the gene normalizers.

    All genes are considered such that also genes may come up that are not reflected Geneformer's vocabulary. This may be fine, since these non-represented genes are likely impacting other, represented, genes.

    Requires a lot of RAM to be able to transpose the sparse matrix (required for efficient computation)
    """

    input:
        read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        gene_normalizers=rules.compute_gene_normalizers.output.gene_mean_log1ps,
        # HTTP.remote("https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_median_dictionary.pkl", keep_local=True)[0],
    output:
        top_genes=PROJECT_DIR / config["paths"]["llava"]["root"] / "top_genes" / "{dataset}.parquet"
    params:
        top_n_genes=100,
    resources:
        mem_mb=500000,
        slurm="cpus-per-task=2"
    conda:
        "cellwhisperer"
    notebook:
        "../notebooks/compute_top_genes.py.ipynb"


rule llava_stage1_dataset:
    """
    Generation of dataset for stage 1 training of LLaVA

    In a nutshell, take the questions (above) and use the previously generated sample annotations as answers
    """
    input:
        annotations_archs4_metasra=ancient(PROJECT_DIR / config["paths"]["processed_multi_annotations"].format(dataset="archs4_metasra")),
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

    TODO: should I force using the 2nd annotation replicate (for archs4_metasra?). I think currently it takes both?? check the output files how it looks
    TODO 2: after generating cellxgene_census samples, run a couple of them through mixtral and convert them to nicer few shot samples

    Few shot input format: JSON (top_gene_sets: list(ranked, avoid scores or ), top_genes: list(ranked), annotation:string, sample_id: string)
    """
    input:
        processed_annotations=PROJECT_DIR / config["paths"]["processed_multi_annotations"],
        gsva=PROJECT_DIR / config["paths"]["gsva"]["result"],
        top_genes=rules.compute_top_genes.output.top_genes,
        request_template=ancient("prompts/llava_stage2_request_template.txt"),
        few_shot_prompts=ancient(expand("prompts/llava_stage2_few_shot_samples/{i}_request.json", i=range(3))),
        few_shot_responses=ancient(expand("prompts/llava_stage2_few_shot_samples/{i}_response.json", i=range(3))),
    output:
        request_splits=scatter.split(PROJECT_DIR / "results/llava/requests/{{dataset,[^/]+}}/{scatteritem}.json"),
        few_shot_block="prompts/few_shot_messages_{dataset}.json",
    params:
        top_n_genes=50,
        top_n_gene_sets=50,
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
        model=Path(config["paths"]["mixtral_model"]).expanduser(),
        instruction="prompts/llava_stage2_few_shot.txt",
        json_split=PROJECT_DIR / "results/llava/requests/{dataset}/{scatteritem}.json",
        json_schema="prompts/llava_stage2_schema.json",
        few_shot_messages="prompts/few_shot_messages_{dataset}.json", # rules.prepare_llava_stage2_requests.output.few_shot_block
    output:
        generated_conversations = protected(PROJECT_DIR / "results" / "llava" / "processed" / "{dataset,[^/]+}" / "{scatteritem}.json"),  # I marked this as protected as it might be costly to produce
    params:
        temperature=0.0,
    resources:
        mem_mb=100000,
        slurm="cpus-per-task=25 gres=gpu:a100-sxm4-80gb:1 qos=a100-sxm4-80gb partition=gpu"
        # slurm="cpus-per-task=25 gres=gpu:a100-sxm4-80gb:1 qos=a100-sxm4-80gb partition=gpu"
    conda: "textgen"  # "../envs/llamacpp.yaml" fails to install :/
    notebook: "../notebooks/llava_stage2_dataset.py.ipynb"


rule generate_llava_complex:
    """
    Stage 2 complex data generation (GPT-4)

    Currently *without* gene names, forcing the model to focus on the provided transcriptome.
    """
    input:
        processed_annotations=PROJECT_DIR / config["paths"]["processed_multi_annotations"],
        gsva=PROJECT_DIR / config["paths"]["gsva"]["result"],
        top_genes=rules.compute_top_genes.output.top_genes,
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

    # TODO check how many failed. go through logs and filter the ones that failed (can we fix them easily?)

    Currently *without* gene names, forcing the model to focus on the provided transcriptome.
    """
    input:
        processed_annotations=PROJECT_DIR / config["paths"]["processed_multi_annotations"],
        gsva=PROJECT_DIR / config["paths"]["gsva"]["result"],
        top_genes=rules.compute_top_genes.output.top_genes,
        system_message=ancient("prompts/llava_stage2_detailed.txt"),
        request_template=ancient("prompts/llava_stage2_request_template.txt"),
    output:
        processed_annotation = protected(PROJECT_DIR / "results" / "llava" / "processed_detailed" / "{dataset}" / "{sample_id}.json"),  # I marked this as protected as it might be costly to produce  # TODO add protected
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
        json_splits = glob.glob("/msc/home/mschae83/cellwhisperer/results/llava/processed/archs4_metasra/second/*-of-128.json"),  # , i=[1, 103, 118, 20, 35, 48, 61, 7, 89, 1, 104, 120, 23, 36, 5, 62, 74, 9, 102, 107, 122, 33, 37, 50, 65, 79, 93])

        # json_splits=[split.format(dataset=dataset)
        #              for dataset in ["archs4_metasra", "cellxgene_census"]
        #              for split in gather.split(PROJECT_DIR / "results" / "llava" / "processed" / "{{dataset}}" / "{scatteritem}.json")
        #              ],
        stage1_train_set = rules.llava_stage1_dataset.output.train_set,
        stage1_test_set = rules.llava_stage1_dataset.output.test_set,
        complex_conversations=ancient(expand(rules.generate_llava_complex.output.processed_annotation, sample_id=COMPLEX_SAMPLES, dataset="archs4_metasra")),
        detailed_conversations=ancient(expand(rules.generate_llava_detailed.output.processed_annotation, sample_id=DETAILED_SAMPLES, dataset="archs4_metasra")),
        transcriptome_weights=PROJECT_DIR / "results/pre_training_processing/archs4_metasra/transcriptome_weights.npz",
        annotation_weights=PROJECT_DIR / "results/pre_training_processing/archs4_metasra/annotation_weights.npz",
    params:
        test_ids=TEST_IDS,
        num_stage1_samples=20000,  # maybe only use 10000. or train with them only in the beginning?
        seed=42,
        transcriptome_tag="<image>",  # we stick to <image> because of the llava code base
        accept_num_erroneous_jsons=200,
    log:
        "logs/aggregate_llava_stage2_dataset.log"
    resources:
        mem_mb=100000,
    output:
        llava_stage2_dataset=PROJECT_DIR / config["paths"]["llava"]["finetune_text_dataset"],
        evaluation_dataset=PROJECT_DIR / config["paths"]["llava"]["evaluation_text_dataset"].format(dataset="archs4_metasra")
    script:
        "../scripts/aggregate_llava_stage2_dataset.py"
