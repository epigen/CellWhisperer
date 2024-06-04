import os
import pandas as pd
import anndata
import subprocess

URL = "https://sharehost.hms.harvard.edu/immgen/GSE227743/GSE227743_Gene_count_table.csv"
TMP_FN = "GSE227743_Gene_count_table.csv"
subprocess.run(f"wget -O {TMP_FN} --no-check-certificate {URL}", shell=True)


df = pd.read_csv(TMP_FN, index_col=0)
df.index.name = "sample_name"
adata = anndata.AnnData(X=df.T, var=df.index.to_frame(), obs=df.columns.to_frame())
adata.obs.index.name = "sample_name"
adata.obs.columns = ["sample_name"]
adata.var.columns = ["gene_name"]
adata.var.index.name = "gene_name"

adata.write_h5ad(snakemake.output.read_count_table)

os.unlink(TMP_FN)
