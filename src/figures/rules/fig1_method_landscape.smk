from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
HTTP = HTTPRemoteProvider()

rule geo_umap_plots:
    """
    """
    input:
        adata=HTTP.remote(f"{config['precomputing_base_url']}/datasets/archs4_geo/cellxgene.h5ad", keep_local=True)[0],
        # adata=PROJECT_DIR / "results" / "archs4_geo" / "{model}" / "cellxgene.h5ad"  # The pipeline could compute it by itself, but it requires a lot of RAM and time
    output:
        cluster_labeled=PROJECT_DIR / config["paths"]["geo_umap"] / "cluster_labeled.svg",
        submission_date_labeled=PROJECT_DIR / config["paths"]["geo_umap"] / "submission_date_labeled.svg",
        highlighted_clusters_date_kdes=PROJECT_DIR / config["paths"]["geo_umap"] / "highlighted_clusters_date_kdes.svg",
    resources:
        mem_mb=300000,  # for good measure
        slurm="cpus-per-task=2"
    params:
        highlight_clusters=["Active Myeloid Differentiation in HSPCs", "K562 Erythroleukemia Cells in Culture", "Obese adipose tissue immune and metabolic state"]  # optional: "Undifferentiated Human Pluripotent Stem cells", "Active Myeloid and T Cell Immune Response",
    conda:
        "cellwhisperer"
    log:
        notebook="../log/geo_umap_plots_{model}.log.ipynb"
    notebook:
        "../notebooks/geo_umap_plots.py.ipynb"
