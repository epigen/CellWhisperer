rule leiden_umap_embeddings:
    input:
        processed_data=rules.process_full_dataset.output.model_outputs,
    output:
        adata=PROJECT_DIR / "results" / "{dataset}" / "leiden_umap_embeddings.h5ad"
    conda:
        "single-cellm"
    log:
        notebook="../log/leiden_umap_embeddings_{dataset}.py.ipynb"
    notebook:
        "../notebooks/leiden_umap_embeddings.py.ipynb"

rule llava_annotate_clusters:
    """
    Needs a 20GB GPU
    """
    input:
        adata=rules.leiden_umap_embeddings.output.adata,
        llava_model=rules.finetune_llava.output.output_dir.format(base_model=LLAVA_BASE_MODEL)
    output:
        adata=PROJECT_DIR / "results" / "{dataset}" / "llava_annotated_clusters.h5ad"
    conda:
        "llava"
    params:
        request="Provide a concise and short description of the sample:",
        num_beams=10
    log:
        notebook="../log/llava_annotate_clusters_{dataset}.py.ipynb"
    notebook:
        "../notebooks/llava_annotate_clusters.py.ipynb"


rule plot_embeddings_with_llava_labels:
    input:
        adata=rules.llava_annotate_clusters.output.adata,
    output:
        **{
            ext: PROJECT_DIR / "results" / "plots" / "plot_dataset_embeddings" / f"{{dataset}}.{ext}"
            for ext in ["png", "pdf", "svg"]
        },
    conda:
        "single-cellm"
    log:
        notebook="../log/plot_embeddings_{dataset}.py.ipynb"
    notebook:
        "../notebooks/plot_embeddings.py.ipynb"
