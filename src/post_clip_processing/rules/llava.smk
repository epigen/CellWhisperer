"""

NOTE this is only for archs4_metasra. Would need to be adapted for cellxgene_census.
"""

# snakemake remote HTTP object
from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
import glob
import pandas as pd


HTTP = HTTPRemoteProvider()

PROJECTOR_TYPE = "mlp2x_8t_gelu"  # TODO fails if > 8 (12 and 16 both failed. either due to  dual-digit not being ok or because they become too large for some reason ). Here is the error evoked (can use pdb to trace (breakpoint before beaks)): /msc/home/mschae83/miniconda3/envs/llava2/lib/python3.10/site-packages/torch/nn/utils/clip_grad.py(55)


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

# random sample (n=100) of the GSVA dataset (which is already weight-corrected)
TEST_IDS = ['SRX8856161', 'SRX2945912', 'SRX12688894', 'SRX7833821',
       'SRX4361085', 'SRX1467354', 'SRX2984546', 'SRX15304075',
       'SRX3364766', 'SRX5215396', 'SRX7652714', 'SRX14745891',
       'SRX4982710', 'SRX6390561', 'SRX15921104', 'SRX1772896',
       'SRX4049419', 'SRX7020537', 'SRX3908039', 'SRX5967356',
       'SRX7749853', 'SRX3161428', 'SRX3650581', 'SRX14286752',
       'SRX7833851', 'SRX13730096', 'SRX3650869', 'SRX4531237',
       'SRX2669558', 'SRX8658385', 'SRX10935235', 'SRX4361797',
       'SRX15139593', 'SRX9959080', 'SRX3806591', 'SRX2810346',
       'SRX11928893', 'SRX15391922', 'SRX4362691', 'SRX4356966',
       'SRX7070031', 'SRX2044754', 'SRX2637359', 'SRX3946791',
       'SRX8816289', 'SRX15919275', 'SRX14745562', 'SRX5496155',
       'SRX1772968', 'SRX5331351', 'SRX7835903', 'SRX15470936',
       'SRX379801', 'SRX8414341', 'SRX3266594', 'SRX9920278',
       'SRX6393360', 'SRX3927543', 'SRX2912916', 'SRX7840513',
       'SRX5178075', 'SRX3802495', 'SRX3806754', 'SRX11146832',
       'SRX2965355', 'SRX17581835', 'SRX5052947', 'SRX7622905',
       'SRX1357859', 'SRX17915534', 'SRX8109658', 'SRX8707050',
       'SRX8455328', 'SRX6682858', 'SRX7549151', 'SRX3606578',
       'SRX8437909', 'SRX18011210', 'SRX3154941', 'SRX5222273',
       'SRX7833468', 'SRX17580837', 'SRX10313523', 'SRX7773354',
       'SRX1301658', 'SRX2325175', 'SRX3806838', 'SRX5216799',
       'SRX1161823', 'SRX1503192', 'SRX3365172', 'SRX5618987',
       'SRX2806190', 'SRX3444884', 'SRX7839000', 'SRX2388408',
       'SRX2491340', 'SRX15036995', 'SRX10659509', 'SRX1789190']

# For efficiency, extract the samples once and store them
if not os.path.exists("./tmp_gsva_samples.csv"):
    GSVA_SAMPLES = pd.read_parquet(PROJECT_DIR / config["paths"]["gsva"]["result"].format(dataset="archs4_metasra")).set_index("Unnamed: 0").drop(columns=["library"]).columns
    # store GSVA samples
    with open("./tmp_gsva_samples.csv", "w") as f:
        f.write("\n".join(GSVA_SAMPLES.to_list()))
else:
    GSVA_SAMPLES = pd.read_csv("./tmp_gsva_samples.csv", header=None).iloc[:, 0]

# make sure they are not part of TEST_IDS or any other set (yet, they should be complementary to the other stage 2 sets. E.g. subsample 15.000 from the 50000 GSVA set)
NUM_COMPLEX_SAMPLES = 5000
NUM_DETAILED_SAMPLES = 10000
CONVERSATION_START = NUM_COMPLEX_SAMPLES + NUM_DETAILED_SAMPLES
COMPLEX_SAMPLES = GSVA_SAMPLES.to_list()[:NUM_COMPLEX_SAMPLES]  # 5000  # list comprehension preserves 'random' order
DETAILED_SAMPLES = GSVA_SAMPLES.to_list()[NUM_COMPLEX_SAMPLES:CONVERSATION_START]
scattergather:
    split=128


rule llava_stage1_dataset:
    """
    Generation of dataset for stage 1 training of LLaVA

    TODO combine single cell and GEO

    In a nutshell, take the questions (above) and use the previously generated sample annotations as answers
    """
    input:
        annotations_archs4_metasra=ancient(PROJECT_DIR / config["paths"]["processed_multi_annotations"].format(dataset="archs4_metasra")),
        # annotations_cellxgene_census=PROJECT_DIR / config["paths"]["processed_annotations"].format(dataset="cellxgene_census"),
    output:
        train_set=PROJECT_DIR / config["paths"]["llava_pretrain_text_dataset"],
        test_set="tmp_output/llava_test_set.json"
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

rule compute_gene_normalizers:
    """
    Compute the gene normalizers (in log scale) for each gene across all samples.

    `np.log(gene + 1).mean()`

    Requires a lot of RAM to be able to transpose the sparse matrix (required for efficient computation)
    """
    input:
        read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
    output:
        gene_mean_log1ps="tmp_output/gene_normalizers/{dataset}.pickle"
    resources:
        mem_mb=500000,
        slurm="cpus-per-task=2"
    conda:
        "cellwhisperer"
    script:
        "../scripts/compute_gene_normalizers.py"

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
        top_genes="tmp_output/top_genes/{dataset}.parquet"
    params:
        top_n_genes=100,
    resources:
        mem_mb=500000,
        slurm="cpus-per-task=2"
    conda:
        "cellwhisperer"
    notebook:
        "../notebooks/compute_top_genes.py.ipynb"

rule prepare_llava_stage2_requests:
    """
    Generate metadata-based requests for the given inputs:

    - Processed annotation
    - GSVA
    - Top-most expressed genes

    Few shot input format: JSON (top_gene_sets: list(ranked, avoid scores or ), top_genes: list(ranked), annotation:string, sample_id: string)
    """
    input:
        processed_annotations=PROJECT_DIR / config["paths"]["processed_multi_annotations"],
        gsva=PROJECT_DIR / config["paths"]["gsva"]["result"],
        top_genes=rules.compute_top_genes.output.top_genes,
        request_template="prompts/llava_stage2_request_template.txt",
        few_shot_prompts=ancient(expand("prompts/llava_stage2_few_shot_samples/{i}_request.json", i=range(3))),
        few_shot_responses=ancient(expand("prompts/llava_stage2_few_shot_samples/{i}_response.json", i=range(3))),
    output:
        request_splits=scatter.split(PROJECT_DIR / "results/post_clip_processing/llava_requests/{{dataset,[^/]+}}/{scatteritem}.json"),
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
        request_template="prompts/llava_stage2_request_template.txt",
        few_shot_prompts=ancient(expand("prompts/llava_stage2_complex_few_shot_samples/{i}_request.json", i=[2, 3, 5])),
        few_shot_responses=ancient(expand("prompts/llava_stage2_complex_few_shot_samples/{i}_response.json", i=[2, 3, 5])),
    output:
        processed_annotation = protected(PROJECT_DIR / "results" / "post_clip_processing" / "llava_processed_complex" / "{dataset}" / "{sample_id}.json"),
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
        request_template="prompts/llava_stage2_request_template.txt",
    output:
        processed_annotation = protected(PROJECT_DIR / "results" / "post_clip_processing" / "llava_processed_detailed" / "{dataset}" / "{sample_id}.json"),  # I marked this as protected as it might be costly to produce  # TODO add protected
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

rule generate_llava_stage2_conversations:
    """
    Generation of dataset for stage 2 training of LLaVA

    This is more sophisticated in that we use Mixtral (or GPT-4) to generate conversations from few-shot examples

    """
    input:
        model=Path(config["paths"]["mixtral_model"]).expanduser(),
        instruction="prompts/llava_stage2_few_shot.txt",
        json_split=PROJECT_DIR / "results/post_clip_processing/llava_requests/{dataset}/{scatteritem}.json",
        json_schema="prompts/llava_stage2_schema.json",
        few_shot_messages="prompts/few_shot_messages_{dataset}.json", # rules.prepare_llava_stage2_requests.output.few_shot_block
    output:
        generated_conversations = protected(PROJECT_DIR / "results" / "post_clip_processing" / "llava_processed" / "{dataset,[^/]+}" / "{scatteritem}.json"),  # I marked this as protected as it might be costly to produce
    params:
        temperature=0.0,
    resources:
        mem_mb=100000,
        slurm="cpus-per-task=25 gres=gpu:a100-sxm4-80gb:1 qos=a100-sxm4-80gb partition=gpu"
        # slurm="cpus-per-task=25 gres=gpu:a100-sxm4-80gb:1 qos=a100-sxm4-80gb partition=gpu"
    conda: "textgen"  # "../envs/llamacpp.yaml" fails to install :/
    notebook: "../notebooks/llava_stage2_dataset.py.ipynb"



rule aggregate_llava_stage2_dataset:
    """
    Read in all the generated annotations and aggregate them into a single JSON file

    NOTE consider oversampling complex items (e.g. use them twice, as they are fewer)
    """
    input:
        json_splits = glob.glob("/msc/home/mschae83/cellwhisperer/results/post_clip_processing/llava_processed/archs4_metasra/second/*-of-128.json"),  # , i=[1, 103, 118, 20, 35, 48, 61, 7, 89, 1, 104, 120, 23, 36, 5, 62, 74, 9, 102, 107, 122, 33, 37, 50, 65, 79, 93])

        # json_splits=[split.format(dataset=dataset)
        #              for dataset in ["archs4_metasra"] # , "cellxgene_census"]  # TODO enable (and make sure to enable the assertion in the script: "assert sample_id not in all_conversations_dict")
        #              for split in gather.split(PROJECT_DIR / "results" / "post_clip_processing" / "llava_processed" / "{{dataset}}" / "{scatteritem}.json")
        #              ],
        stage1_train_set = rules.llava_stage1_dataset.output.train_set,
        stage1_test_set = rules.llava_stage1_dataset.output.test_set,
        complex_conversations=expand(rules.generate_llava_complex.output.processed_annotation, sample_id=COMPLEX_SAMPLES, dataset="archs4_metasra"),
        detailed_conversations=expand(rules.generate_llava_detailed.output.processed_annotation, sample_id=DETAILED_SAMPLES, dataset="archs4_metasra"),
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
        llava_stage2_dataset=PROJECT_DIR / config["paths"]["llava_finetune_text_dataset"],
        evaluation_dataset=PROJECT_DIR / config["paths"]["llava_evaluation_text_dataset"]
    script:
        "../scripts/aggregate_llava_stage2_dataset.py"

rule pretrain_llava:
    """
    Based on /home/moritz/Projects/cellwhisperer/modules/LLaVA/scripts/v1_5/pretrain.sh

    LLaVA takes these image patches as tokens, so a single image leads to ~100(?) text tokens.

    Our transcriptome model returns a single embedding, so to provide more "information power" to the language model, I convert it to 4 tokens.
    Training with `--version plain` leads to deletion of the human prompt (just the transcriptome is passed)

    Original LR is 1e-3. We do 1e-4 because we have slightly more samples (and don't wanto to be too greedy)
    """
    input:
        data_path=rules.llava_stage1_dataset.output[0],
        image_data=rules.process_full_dataset.output.model_outputs.format(dataset=TRAINING_DATASET, model="{model}")  # TODO combine the datasets to be able to train on both
    conda:
        "llava2"
    params:
        deepspeed=True,  # debug if False
        projector_type=PROJECTOR_TYPE,
    output:
        projector=PROJECT_DIR / config["paths"]["llava_pretrained_model_dir"] / "mm_projector.bin",
        output_dir=protected(directory(PROJECT_DIR / config["paths"]["llava_pretrained_model_dir"])),
    resources:
        mem_mb=300000,
        slurm="cpus-per-task=40 gres=gpu:a100-sxm4-80gb:3 qos=a100-sxm4-80gb partition=gpu"
    log:
        "logs/pretrain_llava_{base_model}_{model}.log"
    threads: 16
    shell: """
        PYTHON_SCRIPT=../../modules/LLaVA/llava/train/train_mem.py
        if [[ {params.deepspeed} == True ]]; then
            CMD="deepspeed $PYTHON_SCRIPT --deepspeed ../../modules/LLaVA/scripts/zero2.json"
        else
            CMD="CUDA_LAUNCH_BLOCKING=1 python -m ipdb $PYTHON_SCRIPT"
        fi

        # NOTE for faster debugging try facebook/opt-125m
        $CMD \
            --data_path {input.data_path} \
            --image_data {input.image_data} \
            --output_dir {output.output_dir} \
            --model_name_or_path mistralai/{wildcards.base_model} \
            --version plain \
            --mm_projector_type {params.projector_type} \
            --tune_mm_mlp_adapter True \
            --mm_vision_select_layer -1 \
            --mm_vision_select_feature cls_patch \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --bf16 True \
            --num_train_epochs 1 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 2400 \
            --save_total_limit 1 \
            --learning_rate 1e-4 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --dataloader_num_workers {threads} \
            --report_to wandb \
            --lazy_preprocess True 2>&1| tee {log}
            # --report_to wandb
    """

rule finetune_llava:
    """
    Stage 2 LLaVA training (i.e. fine-tuning LLM). Based on `modules/LLaVA/scripts/v1_5/pretrain.sh`

    # Runs like this on 3+ 80GB GPUs:
    srun -N1 -q a100-sxm4-80gb-sxm4-80gb -c 30 --partition gpu --gres=gpu:a100-sxm4-80gb-sxm4-80gb:4 --mem=200G --pty bash

    TODO: consider curriculum learning: first all of dataset1, then the generated ones
    TODO: consider biomistral (simply assess how well it performs using the perplexity analysis below)
    """
    input:
        data_path=rules.aggregate_llava_stage2_dataset.output[0].format(dataset=TRAINING_DATASET),
        image_data=rules.process_full_dataset.output.model_outputs.format(dataset=TRAINING_DATASET, model="{model}"),
        pretrained_projector=rules.pretrain_llava.output.projector
    conda:
        "llava2"
    params:
        deepspeed=True,
        projector_type=PROJECTOR_TYPE
    output:
        output_dir=protected(directory(PROJECT_DIR / config["paths"]["llava_finetuned_model_dir"])),
    resources:
        mem_mb=300000,
        slurm="cpus-per-task=40 gres=gpu:a100-sxm4-80gb:3 qos=a100-sxm4-80gb partition=gpu"
    log:
        "logs/finetune_llava_{base_model}_{model}.log"
    threads: 16
    shell: """
        PYTHON_SCRIPT=../../modules/LLaVA/llava/train/train_mem.py
        if [[ {params.deepspeed} == True ]]; then
            CMD="deepspeed $PYTHON_SCRIPT --deepspeed ../../modules/LLaVA/scripts/zero3.json"
        else
            CMD="python $PYTHON_SCRIPT"
        fi

        # NOTE for faster debugging try facebook/opt-125m
        $CMD \
            --data_path {input.data_path} \
            --image_data {input.image_data} \
            --output_dir {output.output_dir} \
            --model_name_or_path mistralai/{wildcards.base_model} \
            --version mistral_instruct \
            --pretrain_mm_mlp_adapter {input.pretrained_projector} \
            --mm_projector_type {params.projector_type} \
            --mm_vision_select_layer -1 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --group_by_modality_length False \
            --bf16 True \
            --num_train_epochs 1 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 2400 \
            --save_total_limit 1 \
            --learning_rate 2e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --dataloader_num_workers {threads} \
            --report_to wandb \
            --lazy_preprocess True 2>&1 | tee {log}
            # --report_to wandb
    """

rule llava_evaluation_perplexity:
    """

    """
    input:
        llava_model=ancient(rules.finetune_llava.output.output_dir.format(base_model=LLAVA_BASE_MODEL, model=config["model_name_path_map"]["cellwhisperer"])),
        evaluation_dataset=rules.aggregate_llava_stage2_dataset.output.evaluation_dataset,
        image_data=rules.process_full_dataset.output.model_outputs.format(dataset=TRAINING_DATASET, model=config["model_name_path_map"]["cellwhisperer"]),
    conda:
        "llava2"
    output:
        log_perplexity_ratio=PROJECT_DIR / "results" / "post_clip_processing" / "llava_evaluation_log_mean_perplexity.ratio",  # smaller is better log(ppl_real/ppl_neg_control)
        all_perplexities=PROJECT_DIR / "results" / "post_clip_processing" / "llava_evaluation_all_perplexities.csv",
        comparison_plot=PROJECT_DIR / "results" / "post_clip_processing" / "llava_evaluation_comparison.png"
    params:
        num_projector_tokens=int(PROJECTOR_TYPE.split("_")[1].strip("t"))
    resources:
        mem_mb=300000,
        slurm="cpus-per-task=40 gres=gpu:a100-sxm4-80gb:1 qos=a100-sxm4-80gb partition=gpu"
    log:
        "logs/llava_evaluation_perplexity.log"
    threads: 16
    notebook:
        "../notebooks/llava_evaluation_perplexity.py.ipynb"


rule llava_evaluation_generation:
    """
    Evaluate the generation of the LLaVA model

    # This one here provides an easy going start: /home/moritz/Projects/cellwhisperer/modules/LLaVA/llava/eval/model_vqa.py
    # /home/moritz/Projects/cellwhisperer/modules/LLaVA/llava/eval/model_vqa_loader.py

    # The code fragments stored in the script work (although they are disconnected), but we might want to refrain more to the code in the two files above
    """
    script:
        "../scripts/llava_evaluation_generation.py"

