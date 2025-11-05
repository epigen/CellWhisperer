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
    Uses the `leiden` obs column as well as all columns declared in `.uns["cluster_fields"]` to generate clusters

    Generated CSV has the following columns:
    - cluster_field: obs column name representing clustering (e.g., "leiden")
    - cluster_value: value within the cluster obs column (e.g., leiden cluster ID "1")
    - cluster_annotation: generated annotations
    """
    input:
        embedding_adata=rules.leiden_umap_embeddings.output.adata,
        read_count_adata=PROJECT_DIR / config["paths"]["read_count_table"],  # needed for potential uns and obs fields
        llava_model=ancient(PROJECT_DIR / config["paths"]["llava"]["finetuned_model_dir"].format(llava_dataset="_default", base_model=LLAVA_BASE_MODEL, model="{model}")),
    output:
        csv=PROJECT_DIR / "results" / "{dataset}" / "{model}" / "llava_annotated_clusters.csv"
    conda:
        "llava"
    params:
        request="<s>[INST] Help me analyzing this sample of cells. Respond in proper English sentences in a tone of uncertainty and focus on the biology of the sample rather than any potential donor or patient information (e.g. do not mention age and sex) [/INST] Sure. I'll respond as you requested, focusing on the sample of cells and avoiding patient or donor information. </s> [INST] Describe the biological state of these cells\n<image> [/INST]",
        num_beams=10
    threads: 8  # NOTE increase this without GPU
    resources:
        mem_mb=40000,
        slurm=slurm_gres()
    log:
        notebook="../log/llava_annotate_clusters_{dataset}_{model}.py.ipynb"
    notebook:
        "../notebooks/llava_annotate_clusters.py.ipynb"

rule gpt4_curate_llava_annotations:
    """
    Output is protected to prevent high GPT-4 cost. Script also fails with more than 200 clusters

    NOTE: requires OpenAI API key
    """
    input:
        cellwhisperer_labels=rules.llava_annotate_clusters.output.csv
    output:
        curated_labels=protected(PROJECT_DIR / "results" / "{dataset}" / "{model}" / "llava_curated_annotated_clusters.csv")
    params:
        request="Condense this description of a cell sample into a short title (using normal sentence case) of maximum 8 words. Focus on the biological state of the sample, rather than its source or any specific perturbations.",
        max_num_clusters=200,  # to prevent high GPT-4 cost
        openai_api_key=os.getenv("OPENAI_API_KEY")
    conda:
        "cellwhisperer"
    notebook:
        "../notebooks/gpt4_curate_llava_annotations.py.ipynb"

rule mixtral_curate_llava_annotations:
    """
    """
    input:
        cellwhisperer_labels=rules.llava_annotate_clusters.output.csv,
        model=PROJECT_DIR / config["model_name_path_map"]["mixtral"],
    output:
        curated_labels=PROJECT_DIR / "results" / "{dataset}" / "{model}" / "llava_curated_annotated_clusters_mixtral.csv"
    params:
        request="Condense the description of a cell sample below into a short title (using normal sentence case) of maximum 8 words. Focus on the biological state, rather than the source or any specific perturbations of the sample and generate nothing but the title (no quotes or additional information, using sentence case).\n\n",
    threads: 8  # NOTE increase this without GPU
    resources:
        mem_mb=40000,
        slurm=slurm_gres()
    conda:
        "llama_cpp"
    notebook:
        "../notebooks/mixtral_curate_llava_annotations.py.ipynb"


rule compile_h5ad:
    """
    Compile the generated embeddings and labels into a single h5ad file.

    Also normalizes (log1p) X. If there is a "normalized" layer, it is set alternatively
    """

    input:
        umap_embedding=rules.leiden_umap_embeddings.output.adata,
        cellwhisperer_llava_labels=rules.gpt4_curate_llava_annotations.output.curated_labels if "OPENAI_API_KEY" in os.environ else rules.mixtral_curate_llava_annotations.output.curated_labels,
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
