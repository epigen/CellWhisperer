import scanpy as sc


adata= sc.read_h5ad(snakemake.input.read_count_table)

sampled_indices = (
adata.obs.groupby(snakemake.params.groupby, group_keys=False)
.apply(lambda x: x.sample(min(snakemake.params.max_cells, len(x)), random_state=0))
.index
)

adata = adata[adata.obs.index.isin(sampled_indices)]

adata.write_h5ad(snakemake.output.read_count_table)