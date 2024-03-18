import os
import pandas as pd
import numpy as np
import json
import scanpy as sc
import anndata as ad
import torch
from typing import Tuple

from pathlib import Path

import scipy.sparse as sp
import time
import pickle
import gc

import sys

batch = int(sys.argv[1])

print("batch = " + str(batch) + "\n")


### Setting the dirctories
data_dir = "/scratch/ahakobyan/single_cellm_data/"
project_dir = '/users/anna.hakobyan/projects/single-cellm/'

output_dir = data_dir + "archs4_metasra/gene_ranks/"

os.chdir(project_dir)

### loading the data
# start = time.time()
# exps = sc.read_h5ad(data_dir + "archs4_metasra/full_data_zipped-001.h5ad")
# end = time.time()

# print (end - start) ~10 mins 

### loading the geneformer normalization factors
with open(data_dir + "archs4_metasra/geneformer_gene_median_dictionary.pkl", 'rb') as fp:
     geneformer_gene_medians = pickle.load(fp)


###
# batch = 1

exps = sc.read_h5ad(data_dir + "archs4_metasra/full_data_zipped-001.h5ad")

print("exps have been imported.\n")

gene_indices = [exps.var.gene_name.index.get_loc(key) for key in geneformer_gene_medians.keys() if key in exps.var.gene_name.index]

common_genes = [exps.var.gene_name.index[i] for i in gene_indices]
gene_medians = np.array([geneformer_gene_medians[key] for key in common_genes if key in geneformer_gene_medians])

exps_subset = exps[ (batch-1) * 100000 : min(batch * 100000, exps.X.shape[0]), gene_indices].copy()

library_sizes = np.array(exps.X[(batch-1) * 100000 : min(batch * 100000, exps.X.shape[0]), :].sum(axis = 1)).flatten()

print("GC successful.\n")

gene_ranks = np.zeros(exps_subset.X.shape, dtype = "int32")

target_sum = 1000000

start = time.time()
for i in range(exps_subset.X.shape[1] ):
    normed = sp.csr_matrix(exps_subset.X[i, :] / library_sizes[i] * target_sum / gene_medians)
    gene_ranks[i, :] = np.argsort(normed.toarray() ).flatten()
end = time.time()

print(end - start)

exps_subset.X = gene_ranks


exps_subset.write(output_dir + 'exps_gene_ranks_batch_' + str(batch) +'.h5ad')
# np.savez(data_dir + "gene_ranks.npz", values = gene_ranks)


# # for batch in `seq 5 6`;do  echo $batch; python3 src/experiments/205_llava_tuning_dataset/getting_gene_ranks.py $batch; done

# import scanpy as sc
# import anndata as ad
# import os
# from scipy import sparse

# data_dir = "/scratch/ahakobyan/single_cellm_data/"
# project_dir = '/users/anna.hakobyan/projects/single-cellm/'

# output_dir = data_dir + "archs4_metasra/gene_ranks/"

# os.chdir(project_dir)

# h5ad_files = fnmatch.filter(os.listdir(output_dir), "*_[1-7].h5ad")

# adatas = [ sc.read_h5ad(output_dir + file) for file in h5ad_files]

# concat_out = ad.concat(adatas)

# concat_out.X = sparse.csr_matrix(concat_out.X)

# concat_out.write_h5ad(output_dir + "gene_ranks_csr.h5ad")
