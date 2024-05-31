rule geo_umap_plots:
    """
    """
    input:
        rules.pretraining_processing_compile_h5ad.output.adata.format(dataset="archs4_geo", model="{model}"),
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
