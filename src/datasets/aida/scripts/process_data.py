import anndata

data = anndata.read_h5ad(snakemake.input[0])

# Rename gene_symbol to gene_name (default name in CellWhisperer framework)
data.var.rename(columns={"feature_name": "gene_name"}, inplace=True)

# Move raw counts to the appropriate layer
data.layers["counts"] = data.raw.X.copy()

data.write_h5ad(snakemake.output[0])
