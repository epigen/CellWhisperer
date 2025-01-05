import anndata
import numpy as np
from scipy import sparse
import pickle
from tqdm import tqdm
import concurrent.futures
from cellwhisperer.utils.processing import ensure_raw_counts_adata


# Function to calculate mean log1p for a single gene
def calculate_mean_log1p(column):
    selected = column.squeeze()
    return np.log(selected + 1).mean()


# Read the data
adata = anndata.read_h5ad(snakemake.input.read_count_table)

ensure_raw_counts_adata(adata)

adata.X = sparse.csc_matrix(adata.X)

# Extract the columns (genes) from the sparse matrix
columns = [adata.X[:, i].toarray() for i in tqdm(range(len(tqdm(adata.var))))]

# Calculate the normalization with 0s using parallel processing
gene_mean_log1ps = {}

with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:
    # Map the calculate_mean_log1p function to the columns
    results = list(tqdm(executor.map(calculate_mean_log1p, columns)))

# Combine the results with the gene names
gene_mean_log1ps = dict(zip(adata.var.index, results))

# Save the results
with open(snakemake.output.gene_mean_log1ps, "wb") as f:
    pickle.dump(gene_mean_log1ps, f)
