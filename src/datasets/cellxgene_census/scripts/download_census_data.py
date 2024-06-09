import cellxgene_census

cellxgene_census.download_source_h5ad(
    snakemake.wildcards.dataset_id, to_path=str(snakemake.output[0])
)
