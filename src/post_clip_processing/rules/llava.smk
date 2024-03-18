# snakemake remote HTTP object
from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider

HTTP = HTTPRemoteProvider()

PROJECTOR_TYPE = "mlp2x_8t_gelu"  # TODO fails if > 8 (12 and 16 both failed. either due to  dual-digit not being ok or because they become too large for some reason ). Here is the error evoked (can use pdb to trace (breakpoint before beaks)): /msc/home/mschae83/miniconda3/envs/llava2/lib/python3.10/site-packages/torch/nn/utils/clip_grad.py(55)

QUESTIONS = [
    "What does the sample represent?",
    "What does the transcriptome represent?",
    "Give a brief description of the sample.",
    "Give a brief description of the transcriptome.",
    "Present a compact description of the sample's key features.",
    "Present a compact description of the transcriptome's key features.",
    "Summarize the transcriptomic content of the sample.",
    "Provide a brief description of the given transcriptome.",
    "Provide a brief description of the given sample.",
    "Describe the sample concisely.",
    "Describe the transcriptome concisely.",
]

scattergather:
    split=128


rule llava_stage1_dataset:
    """
    Generation of dataset for stage 1 training of LLaVA

    TODO combine single cell and GEO

    In a nutshell, take the questions (above) and use the previously generated sample annotations as answers
    """
    input:
        annotations_archs4_metasra=PROJECT_DIR / config["paths"]["processed_annotations"].format(dataset="archs4_metasra"),
        # annotations_cellxgene_census=PROJECT_DIR / config["paths"]["processed_annotations"].format(dataset="cellxgene_census"),
    output:
        PROJECT_DIR / config["paths"]["llava_pretrain_text_dataset"]
    params:
        seed=42,
        questions=QUESTIONS,
        transcriptome_tag="<image>",  # we stick to <image> because of the llava code base
        anndata_label_name=config["anndata_label_name"]
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

    TODO: This notebook is currently subject to manual filtering. See https://github.com/epigen/cellwhisperer/issues/339. Check the warning within the notebook
    """
    input:
        processed_annotations=PROJECT_DIR / config["paths"]["processed_annotations"],
        gsva=PROJECT_DIR / config["paths"]["gsva_results"],
        top_genes=rules.compute_top_genes.output.top_genes,
        request_template="prompts/llava_stage2_request_template.txt",
        few_shot_prompts=ancient(expand("prompts/llava_stage2_few_shot_samples/{i}_request.json", i=range(3))),
        few_shot_responses=ancient(expand("prompts/llava_stage2_few_shot_samples/{i}_response.json", i=range(3))),
    output:
        request_splits=scatter.split(PROJECT_DIR / "results/post_clip_processing/llava_requests/{{dataset,[^/]+}}/{scatteritem}.json"),
        few_shot_block="prompts/few_shot_messages_{dataset}.json",
    params:
        top_n_genes=20,
        top_n_gene_sets=20
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
    """
    input:
        json_splits=[split.format(dataset=dataset)
                     for dataset in ["archs4_metasra"] # , "cellxgene_census"]  # TODO enable
                     for split in ["/msc/home/mschae83/cellwhisperer/results/post_clip_processing/llava_processed/{dataset}/1-of-128.json"]
                     # gather.split(PROJECT_DIR / "results" / "post_clip_processing" / "llava_processed" / "{{dataset}}" / "{scatteritem}.json")
                     ]
    output:
        llava_stage2_dataset=PROJECT_DIR / config["paths"]["llava_finetune_text_dataset"],
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
        slurm="cpus-per-task=40 gres=gpu:a100-sxm4-80gb:5 qos=a100-sxm4-80gb partition=gpu"
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
        slurm="cpus-per-task=40 gres=gpu:a100-sxm4-80gb:5 qos=a100-sxm4-80gb partition=gpu"
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
            --version conv_mistral_instruct \
            --pretrain_mm_mlp_adapter {input.pretrained_projector} \
            --mm_projector_type {params.projector_type} \
            --mm_vision_select_layer -1 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --group_by_modality_length True \
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
