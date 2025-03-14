rule prepare_llama31:
    """
    """
    input:
        PROJECT_DIR / config["model_name_path_map"]["llama31orig"]  # downloaded manually
    output:
        PROJECT_DIR / config["model_name_path_map"]["llama31"]
    resources:
        mem_mb=300000
    conda:
        "llava"
    notebook:
        "../notebooks/prepare_llama31.py.ipynb"


rule pretrain_llava:
    """
    Based on /home/moritz/Projects/cellwhisperer/modules/LLaVA/scripts/v1_5/pretrain.sh

    LLaVA takes these image patches as tokens, so a single image leads to ~100(?) text tokens.

    Our transcriptome model returns a single embedding, so to provide more "information power" to the language model, I convert it to 4 tokens.
    Training with `--version plain` leads to deletion of the human prompt (just the transcriptome is passed)

    Original LR is 1e-3. We do 1e-4 because we have slightly more samples (and don't wanto to be too greedy)
    """
    input:
        data_path=rules.llava_stage1_dataset.output["train_set"],
        image_data=rules.combine_processed_data.output.combined,
        base_model_path="/msc/home/mschae83/cellwhisperer_private/resources/{base_model}"
    params:
        deepspeed=True,  # debug if False
        projector_type=config["llava_projector_type"],
        hf_model_name=lambda wildcards: "/msc/home/mschae83/cellwhisperer_private/resources/" + wildcards.base_model, # ("BioMistral/" if "Bio" in wildcards.base_model else "mistralai/") +  wildcards.base_model,  # 
        model_layer_selector=-1
    output:
        projector=PROJECT_DIR / config["paths"]["llava"]["pretrained_model_dir"] / "mm_projector.bin",
        output_dir=protected(directory(PROJECT_DIR / config["paths"]["llava"]["pretrained_model_dir"])),
    resources:
        mem_mb=300000,
        # slurm=slurm_gres(num_gpus=5, num_cpus=40)
        slurm=slurm_gres("medium", num_gpus=3, num_cpus=40)
    conda:
        "llava"
    log:
        "logs/pretrain_llava_{base_model}_{model}.log"
    threads: 16
    shell: """
        PYTHON_SCRIPT=../../modules/LLaVA/llava/train/train_mem.py
        if [[ {params.deepspeed} == True ]]; then
            CMD="deepspeed $PYTHON_SCRIPT --deepspeed ../../modules/LLaVA/scripts/zero2.json"
        else
            export CUDA_LAUNCH_BLOCKING=1
            CMD="python -m pdb $PYTHON_SCRIPT"
        fi

        # NOTE for faster debugging try facebook/opt-125m
        $CMD \
            --data_path {input.data_path} \
            --image_data {input.image_data} \
            --output_dir {output.output_dir} \
            --model_name_or_path {input.base_model_path} \
            --version plain \
            --mm_projector_type {params.projector_type} \
            --tune_mm_mlp_adapter True \
            --mm_vision_select_layer {params.model_layer_selector} \
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
    """

rule finetune_llava:
    """
    Apparently only works on a100-80gb

    Stage 2 LLaVA training (i.e. fine-tuning LLM). Based on `modules/LLaVA/scripts/v1_5/pretrain.sh`

    # Runs like this on 3+ 80GB GPUs:
    srun -N1 -q a100-sxm4-80gb-sxm4-80gb -c 30 --partition gpu --gres=gpu:a100-sxm4-80gb-sxm4-80gb:4 --mem=200G --pty bash

    """
    input:
        data_path=rules.aggregate_llava_stage2_dataset.output["llava_stage2_dataset"],
        image_data=rules.combine_processed_data.output.combined,
        pretrained_projector=rules.pretrain_llava.output.projector,
        base_model_path="/msc/home/mschae83/cellwhisperer_private/resources/{base_model}"
    conda:
        "llava"
    params:
        deepspeed=True,
        projector_type=config["llava_projector_type"],
        hf_model_name=lambda wildcards: "/msc/home/mschae83/cellwhisperer_private/resources/" + wildcards.base_model, #  ("BioMistral/" if "Bio" in wildcards.base_model else "mistralai/") +  wildcards.base_model,
        model_layer_selector=-1
        template_version=lambda wildcards: "llama3_instruct" if "llama-3" in wildcards.base_model.lower() else "mistral_instruct"
    output:
        output_dir=protected(directory(PROJECT_DIR / config["paths"]["llava"]["finetuned_model_dir"])),
    resources:
        mem_mb=300000,
        slurm=slurm_gres("medium", num_gpus=7, num_cpus=40)
        # slurm=slurm_gres(num_gpus=6, num_cpus=40)
    log:
        "logs/finetune_llava_{base_model}_{model}.log"
    threads: 16
    shell: """
        PYTHON_SCRIPT=../../modules/LLaVA/llava/train/train_mem.py
        if [[ {params.deepspeed} == True ]]; then
            CMD="deepspeed $PYTHON_SCRIPT --deepspeed ../../modules/LLaVA/scripts/zero3.json"
        else
            export CUDA_LAUNCH_BLOCKING=1
            CMD="python -m pdb $PYTHON_SCRIPT"
        fi

        # NOTE for faster debugging try facebook/opt-125m
        $CMD \
            --data_path {input.data_path} \
            --image_data {input.image_data} \
            --output_dir {output.output_dir} \
            --model_name_or_path {input.base_model_path} \
            --version {params.template_version} \
            --pretrain_mm_mlp_adapter {input.pretrained_projector} \
            --mm_projector_type {params.projector_type} \
            --mm_vision_select_layer {params.model_layer_selector} \
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
    """
