rule tabsap_celltype_evaluation_dataset:
    input:
        dataset=PROJECT_DIR / config["paths"]["read_count_table"].format(dataset="tabula_sapiens"),
    output:
        evaluation_dataset=PROJECT_DIR / config["paths"]["llava"]["evaluation_text_dataset"].format(dataset="tabula_sapiens"),
    params: 
        celltypes=config["top20_lung_liver_blood_celltypes"],
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
        image_data=rules.process_full_dataset.output.model_outputs.format(dataset="{dataset}", model=config["model_name_path_map"]["cellwhisperer"]),
    conda:
        "llava2"
    output:
        log_perplexity_ratio=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "log_mean_perplexity.ratio",  # smaller is better log(ppl_real/ppl_neg_control)
        all_perplexities=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "all_perplexities.csv",
        comparison_plot=PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "comparison.png"
    params:
        num_projector_tokens=int(PROJECTOR_TYPE.split("_")[1].strip("t")),
        background_shuffle=lambda wildcards: "transcriptome" if wildcards.dataset == "archs4_metasra" else "llm-response",
        num_negatives=30
    resources:
        mem_mb=300000,
        slurm="cpus-per-task=40 gres=gpu:a100-sxm4-80gb:1 qos=a100-sxm4-80gb partition=gpu"
    log:
        "logs/llava_evaluation_perplexity/{dataset}_{base_model}_{model}.log"
    threads: 16
    notebook:
        "../notebooks/llava_evaluation_perplexity.py.ipynb"

# rule llava_evaluation_generation:
#     """
#     Evaluate the generation of the LLaVA model

#     # This one here provides an easy going start: /home/moritz/Projects/cellwhisperer/modules/LLaVA/llava/eval/model_vqa.py
#     # /home/moritz/Projects/cellwhisperer/modules/LLaVA/llava/eval/model_vqa_loader.py

#     # The code fragments stored in the script work (although they are disconnected), but we might want to refrain more to the code in the two files above
#     """
#     script:
#         "../scripts/llava_evaluation_generation.py"
