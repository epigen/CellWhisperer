import anndata
import scanpy as sc
import torch
from scipy import stats
import statsmodels
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from matplotlib_venn import venn2
from collections import defaultdict

from tqdm import tqdm

from cellwhisperer.config import get_path
from cellwhisperer.jointemb.dataset.inference import CellxGenePreparationLoader

import matplotlib
matplotlib.style.use(get_path(["plot_style"]))

sc.set_figure_params(vector_friendly=True, dpi_save=500)

import pickle as pkl

from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.utils.inference import score_modality_vs_texts
from cellwhisperer.config import get_path

from tqdm import tqdm
from joblib import Parallel, delayed

import gseapy as gp


# read ranked_genes
with open(os.path.join(get_path(["uce_paths", "tmp_feature_path"]), "ranked_genes.pkl"), "rb") as f:
    ranked_genes = pkl.load(f)

################################################################################################
# Running prerank

# run prerank on every element in top_genes_list and save the results in a list
prerank_results = []
def run_prerank(i):
    try:
        result = gp.prerank(
            rnk=pd.Series(range(len(ranked_genes[i])), index=ranked_genes[i]),
            gene_sets="GO_Biological_Process_2021",
            organism="Human",
            outdir=None,
            permutation_num=100
        )
        return {i: result}
    except Exception as e:
        print(f"Error in prerank for index {i}: {e}")
        return {i: []}


prerank_results = Parallel(n_jobs=20)(
    delayed(run_prerank)(i) for i in tqdm(range(len(ranked_genes) ), desc="Running Enrichr")
)


#save prerank_results
with open(os.path.join(get_path(["uce_paths", "tmp_feature_path"]), "prerank_results.pkl"), "wb") as f:
    pkl.dump(prerank_results, f)




# read top_genes_list list
with open(os.path.join(get_path(["uce_paths", "tmp_feature_path"]), "top_genes_list.pkl"), "rb") as f:
    top_genes_list = pkl.load(f)


enrich_results = []

def run_enrich(i):
    try:
        result = gp.enrichr(
        gene_list=list(top_genes_list[i]),
        gene_sets=["GO_Biological_Process_2021"],
        organism="Human",
        outdir=f"test/enr_DEGs_GOBP_{i}",
        cutoff=0.1,
    )
        return result
    except Exception as e:
        print(f"Error in prerank for index {i}: {e}")
        return []
    

enrich_results = Parallel(n_jobs=20)(
    delayed(run_enrich)(i) for i in tqdm(range(len(top_genes_list) ), desc="Running Enrichr")
)


#save enrich_results
with open(os.path.join(get_path(["uce_paths", "tmp_feature_path"]), "enrich_results.pkl"), "wb") as f:
    pkl.dump(enrich_results, f)

