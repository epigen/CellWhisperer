PROJECTOR_TYPE = "mlp2x_4t_gelu"

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

rule create_text_dataset:
    input:
        PROJECT_DIR / config["paths"]["full_dataset"]
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
        "../notebooks/create_text_dataset.py.ipynb"

rule pretrain_llava:
    """
    Based on /home/moritz/Projects/cellwhisperer/modules/LLaVA/scripts/v1_5/pretrain.sh

    LLaVA takes these image patches as tokens, so a single image leads to ~100(?) text tokens.
    Our transcriptome model returns a single embedding, so to provide more "information power" to the language model, I convert it to 4 tokens.
    """
    input:
        data_path=rules.create_text_dataset.output[0].format(dataset=TRAINING_DATASET),
        image_data=rules.process_full_dataset.output.model_outputs.format(dataset=TRAINING_DATASET, model="{model}")
    conda:
        "llava"
    params:
        deepspeed=True,
        projector_type=PROJECTOR_TYPE,
    output:
        projector=PROJECT_DIR / config["paths"]["llava_pretrained_model_dir"] / "mm_projector.bin",
        output_dir=protected(directory(PROJECT_DIR / config["paths"]["llava_pretrained_model_dir"])),
    resources:
        mem_mb=100000,
        slurm="cpus-per-task=20 gres=gpu:a100-sxm4-80gb:4 qos=a100-sxm4-80gb partition=gpu"
    log:
        "logs/pretrain_llava_{base_model}_{model}.log"
    threads: 16
    shell: """
        PYTHON_SCRIPT=../../modules/LLaVA/llava/train/train_mem.py
        if [[ {params.deepspeed} == True ]]; then
            CMD="deepspeed $PYTHON_SCRIPT --deepspeed ../../modules/LLaVA/scripts/zero2.json"
        else
            CMD="python $PYTHON_SCRIPT"
        fi

        # NOTE for faster debugging try facebook/opt-125m
        $CMD \
            --data_path {input.data_path} \
            --image_data {input.image_data} \
            --output_dir {output.output_dir} \
            --model_name_or_path lmsys/{wildcards.base_model} \
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
            --learning_rate 1e-3 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --dataloader_num_workers {threads} \
            --lazy_preprocess True 2>&1| tee {log}
            # --report_to wandb
    """

rule finetune_llava:
    """
    Based on /home/moritz/Projects/cellwhisperer/modules/LLaVA/scripts/v1_5/pretrain.sh

    # Runs like this on 3 80GB GPUs:
    srun -N1 -q a100-sxm4-80gb -c 30 --partition gpu --gres=gpu:a100-sxm4-80gb:4 --mem=200G --pty bash

    # TODO consider decreasing learning rate (in function of the number of training samples)
    """
    input:
        data_path=rules.create_text_dataset.output[0].format(dataset=TRAINING_DATASET),
        image_data=rules.process_full_dataset.output.model_outputs.format(dataset=TRAINING_DATASET, model="{model}"),
        pretrained_projector=rules.pretrain_llava.output.projector
    conda:
        "llava"
    params:
        deepspeed=True,
        projector_type=PROJECTOR_TYPE
    output:
        output_dir=protected(directory(PROJECT_DIR / config["paths"]["llava_finetuned_model_dir"])),
    resources:
        mem_mb=100000,
        slurm="cpus-per-task=20 gres=gpu:a100-sxm4-80gb:4 qos=a100-sxm4-80gb partition=gpu"
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
            --model_name_or_path lmsys/{wildcards.base_model} \
            --version v1 \
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
            --lazy_preprocess True 2>&1 | tee {log}
            # --report_to wandb
    """
