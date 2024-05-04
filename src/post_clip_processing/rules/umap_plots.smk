rule geo_umap_plots:
    """
    """
    input:
        rules.compile_h5ad.output.adata.format(dataset="archs4_metasra", model="{model}"),
    output:
        cluster_labeled=PROJECT_DIR / config["paths"]["geo_umap"] / "cluster_labeled.svg",
        submission_date_labeled=PROJECT_DIR / config["paths"]["geo_umap"] / "submission_date_labeled.svg",
        highlighted_clusters_date_kdes=PROJECT_DIR / config["paths"]["geo_umap"] / "highlighted_clusters_date_kdes.svg",
    params:
        highlight_clusters=["Active Myeloid Differentiation in HSPCs", "K562 Erythroleukemia Cells in Culture", "Obese adipose tissue immune and metabolic state"]  # optional: "Undifferentiated Human Pluripotent Stem cells", "Active Myeloid and T Cell Immune Response",

    conda:
        "cellwhisperer"
    log:
        notebook="../log/geo_umap_plots_{model}.log.ipynb"
    notebook:
        "../notebooks/geo_umap_plots.py.ipynb"


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
