import anndata
import scanpy as sc


adata = anndata.read_csv(
    snakemake.input[0],
    first_column_names=True,
).T


# TODO: Hardcoded metadata extraction, specific for immgen
adata.obs["cell type"] = [x.split("#")[0] for x in adata.obs.index.values]
# TODO: Hardcoded metadata extraction, specific for immgen
adata.obs["cell type rough"] = [x.split(".")[0] for x in adata.obs["cell type"].values] 
annot = sc.queries.biomart_annotations(
    "hsapiens",
    ["ensembl_gene_id", "external_gene_name"],
).set_index("external_gene_name")
annot_drop_dups = annot.reset_index().drop_duplicates(subset="external_gene_name")
annot_drop_dups = annot_drop_dups.set_index("external_gene_name")
adata_w_id = adata[:, [x for x in adata.var.index if x in annot.index]]
adata_w_id.var["ensembl_id"] = annot_drop_dups.loc[
    adata_w_id.var.index.values, "ensembl_gene_id"
].values
sc.pp.calculate_qc_metrics(adata_w_id, inplace=True)
adata_w_id.obs["n_counts"] = adata_w_id.obs.total_counts
adata_w_id.obs.index.name = "sample_name"
adata_w_id.obs.reset_index(inplace=True)
adata_w_id.write_loom(snakemake.output[0])