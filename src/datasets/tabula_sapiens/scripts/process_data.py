import anndata

data = anndata.read_h5ad(snakemake.input[0])

# Rename gene_symbol to gene_name (default name in CellWhisperer framework)
data.var.rename(columns={"gene_symbol": "gene_name"}, inplace=True)

# Set the raw counts as the default layer
data.layers["normalized"] = data.X
data.X = data.layers["raw_counts"].copy()

# Remove raw counts layer to save space
del data.layers["raw_counts"]

# For convenience, set the ensembl id as index
data.var.set_index("ensemblid", inplace=True)

# save structured annotations
data.obs.to_json(snakemake.output.structured_annotations_full)


data.write_h5ad(snakemake.output[0])
