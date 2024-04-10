import anndata
import numpy as np
from scipy import sparse
from tqdm.auto import tqdm
import pickle


adata = anndata.read_h5ad(snakemake.input.read_count_table)
adata.X = sparse.csc_matrix(adata.X)

# Calculate the normalization with 0s

gene_mean_log1ps = {}

for var in tqdm(adata.var.index):
    selected = adata[:, var].X.toarray().squeeze()

    gene_mean_log1ps[var] = np.log(selected + 1).mean()

with open(snakemake.output.gene_mean_log1ps, "wb") as f:
    pickle.dump(gene_mean_log1ps, f)
