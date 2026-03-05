import os

rule leiden_umap_embeddings:
    input:
        processed_data=PROJECT_DIR / config["paths"]["model_processed_dataset"],
    output:
        adata=PROJECT_DIR / "results" / "{dataset}" / "{model}" / "leiden_umap_embeddings.h5ad"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=240000,
        slurm="cpus-per-task=5 qos=cpu partition=cpu"
    log:
        notebook="../log/leiden_umap_embeddings_{dataset}_{model}.py.ipynb"
    notebook:
        "../notebooks/leiden_umap_embeddings.py.ipynb"

rule llava_annotate_clusters:
    """
    Uses the hosted CellWhisperer API to annotate clusters.

    Generated CSV has the following columns:
    - cluster_field: obs column name representing clustering (e.g., "leiden")
    - cluster_value: value within the cluster obs column (e.g., leiden cluster ID "1")
    - cluster_annotation: generated annotations
    """
    input:
        embedding_adata=rules.leiden_umap_embeddings.output.adata,
        read_count_adata=PROJECT_DIR / config["paths"]["read_count_table"],
    output:
        csv=PROJECT_DIR / "results" / "{dataset}" / "{model}" / "llava_annotated_clusters.csv"
    params:
        api_url=config["cellwhisperer_api_url"] + "/llava-model-worker",
        model_name="Mistral-7B-Instruct-v0.2__cellwhisperer_clip_v1",
    conda:
        "cellwhisperer"
    log:
        notebook="../log/llava_annotate_clusters_{dataset}_{model}.py.ipynb"
    notebook:
        "../notebooks/llava_annotate_clusters_api.py.ipynb"

rule curate_llava_annotations:
    """
    Condenses LLaVA cluster descriptions into short (<=8 word) titles.
    Uses OpenAI GPT-4 if OPENAI_API_KEY is set and valid, otherwise falls back
    to a local model (Qwen2.5-3B-Instruct) via transformers.
    """
    input:
        cellwhisperer_labels=rules.llava_annotate_clusters.output.csv
    output:
        curated_labels=PROJECT_DIR / "results" / "{dataset}" / "{model}" / "llava_curated_annotated_clusters.csv"
    params:
        request="Condense this description of a cell sample into a short title (using normal sentence case) of maximum 8 words. Focus on the biological state of the sample, rather than its source or any specific perturbations. Generate nothing but the title.",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        local_model="Qwen/Qwen2.5-0.5B-Instruct",
    resources:
        mem_mb=16000,
    conda:
        "cellwhisperer"
    log:
        notebook="../log/curate_llava_annotations_{dataset}_{model}.py.ipynb"
    notebook:
        "../notebooks/curate_llava_annotations.py.ipynb"


rule compile_h5ad:
    """
    Compile the generated embeddings and labels into a single h5ad file.

    Also normalizes (log1p) X. If there is a "normalized" layer, it is set alternatively
    """

    input:
        umap_embedding=rules.leiden_umap_embeddings.output.adata,
        cellwhisperer_llava_labels=rules.curate_llava_annotations.output.curated_labels,
        read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        processed_data=PROJECT_DIR / config["paths"]["model_processed_dataset"], # rules.process_full_dataset.output.model_outputs,
        enrichr_terms=PROJECT_DIR / config["paths"]["enrichr_terms_json"],
        gene_log1p_normalizers=PROJECT_DIR / "results" / "gene_normalizers" / "{dataset}.pickle",  # NOTE not required anymore actually
        top_genes=PROJECT_DIR / "results" / "{dataset}" / "top_genes.parquet",
    output:
        adata=PROJECT_DIR / "results" / "{dataset}" / "{model}" / "cellxgene.h5ad"
    params:
        max_categories_filter=500
    resources:
        mem_mb=1000000,
        slurm="cpus-per-task=2"
    conda:
        "cellwhisperer"
    log:
        notebook="../log/compile_h5ad_{dataset}_{model}.py.ipynb"
    notebook:
        "../notebooks/compile_h5ad.py.ipynb"
