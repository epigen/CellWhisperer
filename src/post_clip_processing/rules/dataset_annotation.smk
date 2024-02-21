rule leiden_umap_embeddings:
    input:
        processed_data=rules.process_full_dataset.output.model_outputs,
    output:
        adata=PROJECT_DIR / "results" / "{dataset}" / "{model}" / "leiden_umap_embeddings.h5ad"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=24000,
        slurm="cpus-per-task=5 qos=cpu partition=cpu"
    log:
        notebook="../log/leiden_umap_embeddings_{dataset}_{model}.py.ipynb"
    notebook:
        "../notebooks/leiden_umap_embeddings.py.ipynb"

rule llava_annotate_clusters:
    """

    Generated CSV has two cols. (1) leiden cluster ID. (2) annotation
    TODO need to change code to not produce the h5ad
    """
    input:
        adata=rules.leiden_umap_embeddings.output.adata,
        llava_model=ancient(rules.finetune_llava.output.output_dir.format(base_model=LLAVA_BASE_MODEL, model="{model}")),
    output:
        csv=PROJECT_DIR / "results" / "{dataset}" / "{model}" / "llava_annotated_clusters.csv"
    conda:
        "llava"
    params:
        request="Provide a concise and short description of the sample:",  # TODO try keyword
        num_beams=10
    resources:
        mem_mb=40000,
        slurm="cpus-per-task=5 gres=gpu:a100:1 qos=a100 partition=gpu"
    log:
        notebook="../log/llava_annotate_clusters_{dataset}_{model}.py.ipynb"
    notebook:
        "../notebooks/llava_annotate_clusters.py.ipynb"

rule cellwhisperer_annotate_clusters:
    """
    Needs a 5GB GPU

    """
    input:
        adata=rules.leiden_umap_embeddings.output.adata,
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
    output:
        csv=PROJECT_DIR / "results" / "{dataset}" / "{model}" / "cellwhisperer_annotated_clusters.csv",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=40000,
        slurm="cpus-per-task=5 gres=gpu:a100:1 qos=a100 partition=gpu"
    log:
        notebook="../log/cellwhisperer_annotate_clusters_{dataset}_{model}.py.ipynb"
    notebook:
        "../notebooks/cellwhisperer_annotate_clusters.py.ipynb"

rule gpt4_curate_cluster_keywords:
    """
    Output is protected to prevent high GPT-4 cost. Script also fails with more than 200 clusters
    TODO: my key is stored in here. needs to be provided as environment variable
    """
    input:
        cellwhisperer_labels=rules.cellwhisperer_annotate_clusters.output.csv,
    output:
        curated_labels=protected(PROJECT_DIR / "results" / "{dataset}" / "{model}" / "cellwhisperer_curated_annotated_clusters.csv")
    params:
        max_num_clusters=200  # to prevent high GPT-4 cost
    conda:
        "cellwhisperer"
    notebook:
        "../notebooks/gpt4_curate_cluster_keywords.py.ipynb"


rule compile_h5ad:
    """
    Compile the generated embeddings and labels into a single h5ad file

    Also normalizes (log1p) X. If there is a "normalized" layer, it is set alternatively
    """

    input:
        # llava_labels=rules.llava_annotate_clusters.output.csv,  # NOTE: include this once the llava-approach becomes powerful enough
        umap_embedding=rules.leiden_umap_embeddings.output.adata,
        cellwhisperer_labels=rules.gpt4_curate_cluster_keywords.output.curated_labels,
        full_data=PROJECT_DIR / config["paths"]["full_dataset"],
        processed_data=PROJECT_DIR / config["paths"]["model_processed_dataset"], # rules.process_full_dataset.output.model_outputs,
        enrichr_terms=PROJECT_DIR / config["paths"]["enrichr_terms_json"],
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
    output:
        adata=PROJECT_DIR / "results" / "{dataset}" / "{model}" / "cellxgene.h5ad"
    params:
        max_categories_filter=500
    conda:
        "cellwhisperer"
    log:
        notebook="../log/compile_h5ad_{dataset}_{model}.py.ipynb"
    notebook:
        "../notebooks/compile_h5ad.py.ipynb"


# rule plot_embeddings_with_llava_labels:
#     """
#     Plot the embeddings with the llava labels
#     TODO this is potentially broken because it was implemented for another adata
#     TODO code is here https://github.com/epigen/cellwhisperer/issues/234#issuecomment-1919533112 (adopted from [[id:8d3f5470-f4d2-4c1b-9572-40305bd62073][24. Fig 1b: Color the embeddings by interesting metrics · Issue #234 · epigen/cellwhisperer]])
#     """
#     input:
#         adata=rules.compile_h5ad.output.adata,
#     output:
#         **{
#             ext: PROJECT_DIR / "results" / "plots" / "plot_dataset_embeddings" / f"{{dataset}}_{{model}}.{ext}"
#             for ext in ["png", "pdf", "svg"]
#         },
#     resources:
#         mem_mb=24000,
#         slurm="cpus-per-task=5 qos=cpu partition=cpu"
#     conda:
#         "cellwhisperer"
#     log:
#         notebook="../log/plot_embeddings_{dataset}_{model}.py.ipynb"
#     notebook:
#         "../notebooks/plot_embeddings.py.ipynb"
