"""
We could honestly also compute everything directly within the {dataset}.h5ad. However, this would lead to 3x recreation of that massive file. A single job could be another solution, but the we would need scanpy in the llava environment. problems over problems.

Hmm. We anyways need to integrate llava into single_cellm! Going with a single env (single-cellm) would also fix the notebooks issue!
"""

rule leiden_umap_embeddings:
    input:
        processed_data=rules.process_full_dataset.output.model_outputs,
    output:
        adata=PROJECT_DIR / "results" / "{dataset}" / "{model}" / "leiden_umap_embeddings.h5ad"
    conda:
        "single-cellm"
    resources:
        mem_mb=24000,
        slurm="cpus-per-task=5 qos=cpu partition=cpu"
    log:
        notebook="../log/leiden_umap_embeddings_{dataset}_{model}.py.ipynb"
    notebook:
        "../notebooks/leiden_umap_embeddings.py.ipynb"

rule llava_annotate_clusters:
    """
    Needs a 20GB GPU
    """
    input:
        adata=rules.leiden_umap_embeddings.output.adata,
        llava_model=ancient(rules.finetune_llava.output.output_dir.format(base_model=LLAVA_BASE_MODEL, model="{model}")),
    output:
        adata=PROJECT_DIR / "results" / "{dataset}" / "{model}" / "llava_annotated_clusters.h5ad"
    conda:
        "llava"
    params:
        request="Provide a concise and short description of the sample:",  # TODO try keyword
        num_beams=10
    resources:
        mem_mb=40000,
        slurm="cpus-per-task=5 gres=gpu:3g.20gb:1 qos=a100 partition=gpu"
    log:
        notebook="../log/llava_annotate_clusters_{dataset}_{model}.py.ipynb"
    notebook:
        "../notebooks/llava_annotate_clusters.py.ipynb"

rule single_cellm_annotate_clusters:
    input:
        adata=rules.leiden_umap_embeddings.output.adata,
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
    output:
        csv=PROJECT_DIR / "results" / "{dataset}" / "{model}" / "single_cellm_annotated_clusters.csv",
    conda:
        "single-cellm"
    log:
        notebook="../log/single_cellm_annotate_clusters_{dataset}_{model}.py.ipynb"
    notebook:
        "../notebooks/single_cellm_annotate_clusters.py.ipynb"

rule gpt4_curate_cluster_keywords:
    """
    TODO Integrate file /home/moritz/wiki/roam/24_fig_1b_color_the_embeddings_by_interesting_metrics_issue_234_epigen_single_cellm.org for this processing step
    TODO then, intgerate it into compile_h5ad
    A generated version is here: 
    https://drive.google.com/drive/folders/1y6gmv0Z19mW-2-S7Et22-raZopqNg7SC?usp=drive_link
    """
    input:
        single_cellm_labels=rules.single_cellm_annotate_clusters.output.csv,
    output:
        curated_labels=PROJECT_DIR / "results" / "{dataset}" / "{model}" / "single_cellm_curated_annotated_clusters.csv"
    # conda:
    #     "single-cellm"
    shell: ""


rule compile_h5ad:
    """
    Compile the generated embeddings and labels into a single h5ad file

    Also normalizes (log1p) X. If there is a "normalized" layer, it is set alternatively

    """

    input:
        llava_labels=rules.llava_annotate_clusters.output.adata,  # TODO use CSV in future and take rules.leiden_umap_embeddings.output.adata as additional input
        single_cellm_labels=rules.gpt4_curate_cluster_keywords.output.curated_labels,
        full_data=PROJECT_DIR / config["paths"]["full_dataset"],
        processed_data=PROJECT_DIR / config["paths"]["model_processed_dataset"], # rules.process_full_dataset.output.model_outputs,
        enrichr_terms=PROJECT_DIR / config["paths"]["enrichr_terms_json"],
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
    output:
        adata=PROJECT_DIR / "results" / "{dataset}" / "{model}" / "cellxgene.h5ad"
    params:
        max_categories_filter=500
    conda:
        "single-cellm"
    log:
        notebook="../log/compile_h5ad_{dataset}_{model}.py.ipynb"
    notebook:
        "../notebooks/compile_h5ad.py.ipynb"


rule plot_embeddings_with_llava_labels:
    """
    Plot the embeddings with the llava labels
    TODO this is potentially broken because it was implemented for another adata
    TODO code is here https://github.com/epigen/single-cellm/issues/234#issuecomment-1919533112 (adopted from [[id:8d3f5470-f4d2-4c1b-9572-40305bd62073][24. Fig 1b: Color the embeddings by interesting metrics · Issue #234 · epigen/single-cellm]])
    """
    input:
        adata=rules.compile_h5ad.output.adata,
    output:
        **{
            ext: PROJECT_DIR / "results" / "plots" / "plot_dataset_embeddings" / f"{{dataset}}_{{model}}.{ext}"
            for ext in ["png", "pdf", "svg"]
        },
    resources:
        mem_mb=24000,
        slurm="cpus-per-task=5 qos=cpu partition=cpu"
    conda:
        "single-cellm"
    log:
        notebook="../log/plot_embeddings_{dataset}_{model}.py.ipynb"
    notebook:
        "../notebooks/plot_embeddings.py.ipynb"
