# %% [markdown]
# ### Preparation

# %%
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

import pickle as pkl

import matplotlib
matplotlib.style.use(get_path(["plot_style"]))

import gseapy as gp

sc.set_figure_params(vector_friendly=True, dpi_save=500)

# %%
# Cellwhisperer scoring
from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.utils.inference import score_transcriptomes_vs_texts
from cellwhisperer.config import get_path


# %%
ckpt_file_path = "./results/models/jointemb/cellwhisperer_clip_v2_uce.ckpt"
(   pl_model_cellwhisperer,
    text_processor_cellwhisperer,
    transcriptome_processor_cellwhisperer,
) = load_cellwhisperer_model(model_path=ckpt_file_path, eval=True)


# %%
texts_list = ["CD4+ T cell", "muscle"]

# %% [markdown]
# ### Loading the subsample scores

# %%
SUBSAMPLE_ARCHS4_DATA_PATH = "./data/archs4_geo/cellxgene_subsampled_01.h5ad"
sub_archs4_adata = anndata.read_h5ad(SUBSAMPLE_ARCHS4_DATA_PATH)

# %%
import pickle
import os

filename = "./data/archs4_geo/cellxgene_subsampled_01_scores.pkl"

if os.path.exists(filename) and os.path.getsize(filename) > 0:
    with open(filename, "rb") as f:
        subsample_score = pickle.load(f)
else:
    raise FileNotFoundError(f"The file '{filename}' is either missing or empty.")

# %%
# %load_ext autoreload
# %autoreload 2

# %%
sub_archs4_adata.layers["counts"] = sub_archs4_adata.X.copy()

# %% [markdown]
# ### Subsampling the subsample

# %%
# make a scatterplot with scores
plt.figure(figsize=(5, 5))

# Ensure that the cluster column is converted to a categorical type if it's not already
# adata.obs["cluster"] = adata.obs["leiden"].astype('category')


# %%
sub_archs4_adata.obsm["transcriptome_embeds"].shape

# %%
# get cell idxs with CD4+ T cell score with a score over 7 (arbitrary threshold)
cd4_cell_idx = np.where(subsample_score[0, :] > 3.5)[0]
print("number of selected CD4+ T cells:{}".format(len(cd4_cell_idx)) )

# get cells idxs with muscle index with a score over 7 (arbitrary threshold)
muscle_cell_idx = np.where(subsample_score[1, :] > 3.5)[0]
print("number of selected muscle cells:{}".format(len(muscle_cell_idx)) )


# get random 1000 cells from the dataset
np.random.seed(42)
random_idx = np.random.choice(sub_archs4_adata.obsm["transcriptome_embeds"].shape[0], 1000, replace=False)

# combine cd4 and muscle cell and random cell idxs
selected_cells_idx = np.concatenate([cd4_cell_idx, muscle_cell_idx, random_idx])
print("number of selected cells:{}".format(len(selected_cells_idx)) )

# %%
adata_subset = sub_archs4_adata[selected_cells_idx, :]
scores_subset = subsample_score[:, selected_cells_idx]

# %% [markdown]
# ### Creating the dataloader

# %%
def ensure_raw_counts_adata(adata):
    # Check if the values in the X layer are counts (i.e., integers)
    comp = np.abs(adata.X[:100] - adata.X[:100].astype(int))
    if isinstance(adata.X, sparse.csr_matrix):
        comp = comp.toarray()

    if not np.all(comp < 1e-6):
        try:
            adata.X = adata.layers["counts"]
        except KeyError:
            raise ValueError(
                'adata.X contains normalized counts, but raw counts are not provided in adata.layers["counts"].'
            )


# %%
from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader
from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor
from cellwhisperer.jointemb.dataset.jointemb import JointEmbedDataset
class CellxGenePreparationLoader(DataLoader):
    """
    Prepare the dataset.

    Only prepare the transcriptome data
    """

    def __init__(
        self,
        read_count_table: Union[anndata.AnnData, Path, str],
        transcriptome_processor="geneformer",
        transcriptome_processor_kwargs={},
        **kwargs
    ):
        """ """
        if isinstance(read_count_table, (str, Path)):
            read_count_table = anndata.read_h5ad(read_count_table)

        # ensure_raw_counts_adata(read_count_table)

        self.transcriptome_processor = transcriptome_processor
        self.transcriptome_processor_kwargs = transcriptome_processor_kwargs

        # Load data and processor
        processor = TranscriptomeTextDualEncoderProcessor(
            self.transcriptome_processor,
            "dmis-lab/biobert-v1.1",  # unused
        )

        inputs = processor(
            text=None,
            transcriptomes=read_count_table,
            return_tensors="pt",
            padding="max_length",  # not sure if required (shouldn't actually)
        )

        dataset = JointEmbedDataset(
            inputs,
            orig_ids=read_count_table.obs.index.to_numpy(),
        )
        super().__init__(dataset, **kwargs)


# %% [markdown]
# ### Captum

# %%
uce_model = pl_model_cellwhisperer.model.transcriptome_model.model
pl_model_cellwhisperer.model.transcriptome_model = pl_model_cellwhisperer.model.transcriptome_model.model.eval().to(
    pl_model_cellwhisperer.model.device
) # TODO transcriptome model shouldn't be frozen actually..

# %%
import captum

# %%
# _, cls = uce_model.forward(**transcriptome_batch)
# cls
# cls.shape
# del cls

# %%
query_batch = text_processor_cellwhisperer(["CD4+ T cell"])
query_batch 

# %%
query_features, query_embeds = pl_model_cellwhisperer.model.get_text_features(**{k: torch.tensor(t).to(pl_model_cellwhisperer.model.device) for k,t in query_batch.items()}, normalize_embeds=True)
query_embeds = query_embeds.detach() # treat query as constant value to 'optimize against'
query_embeds.shape


# %%
import torch.nn as nn
# TODO Might be implemented directly via my `model.py` (just reuse the `forward` function?). That way we also stay flexible with query_embeds
class IntegrationModule(nn.Module):
    def __init__(self, pl_model_cellwhisperer, query_embeds):  
        super(IntegrationModule, self).__init__()
        self.model = pl_model_cellwhisperer.model
        self.query_embeds = query_embeds

    def forward(self, expression_expr, expression_key_padding_mask):
        transcriptome_embeds = self.model.get_transcriptome_features(
            expression_expr=expression_expr,
            expression_key_padding_mask=expression_key_padding_mask,
            normalize_embeds=True
        )[1]
        
        res = torch.einsum("nd,md->nm", [self.query_embeds, transcriptome_embeds]) * \
              self.model.discriminator.temperature.exp().detach()
        return res[0]

# %%
integrator = IntegrationModule(pl_model_cellwhisperer, query_embeds)
# integrator(transcriptome_batch["expression_expr"], transcriptome_batch["expression_key_padding_mask"])

# %% [markdown]
# ### Code to return from transcriptome embeddings to genes

# %%
# ls -lht /nobackup/lab_cresswell/ahakobyan/cellwhisperer_private/results/UCE | head  

# %%
from UCE.data_proc.data_utils import (
    get_species_to_pe,
    anndata_to_sc_dataset,
    data_to_torch_X,
    get_spec_chrom_csv,
    adata_path_to_prot_chrom_starts,
)

# %%
# pad_length=1152
# sample_size = 1024

# TOKEN_DIM = 5120
# PE_DIM = 1280  # ESM2 embedding dimension

# cls_token_idx = 3
# chrom_token_offset = 143574
# chrom_token_right_idx = 2
# pad_token_idx = 0

# %% [markdown]
# ### Going from embeds to genes

# %%
# Suppress scientific notation for tensor printing
torch.set_printoptions(sci_mode=False)

# %%


# %% [markdown]
# ### Attributing importance to all cells

# %%
dl = CellxGenePreparationLoader(
    read_count_table=adata_subset,
    transcriptome_processor=pl_model_cellwhisperer.model.transcriptome_model.config.model_type,
    batch_size=1,  # low for testing
)

# %%
# clean up cuda memory
torch.cuda.empty_cache()

# %%
torch.cuda.ipc_collect()

# %%
from tqdm import tqdm


# %%
device = pl_model_cellwhisperer.model.device
all_res_list = []

for batch_idx, transcriptome_batch in enumerate(tqdm(dl, desc="Processing batches")):
    ig = captum.attr.LayerIntegratedGradients(
        integrator, 
        pl_model_cellwhisperer.model.transcriptome_model.uce_model.pe_embedding
    )
    res = ig.attribute(
        inputs=transcriptome_batch["expression_expr"].to(device),
        # baselines=cell_features_baseline,  # TODO 
        additional_forward_args=(transcriptome_batch["expression_key_padding_mask"].to(device).detach(), ),
        internal_batch_size=1,  # already needs >40GB of VRAM
        n_steps=20,  # TODO increase for higher accuracy
    )
    all_res_list.append(res.sum(-1).argsort())

all_res = torch.cat(all_res_list, dim=0)

# %%
# Ensure `get_path` returns a valid path string
output_path = os.path.join(get_path(["uce_paths", "tmp_feature_path"]), "all_res_list_20steps.torch")
torch.save(all_res_list, output_path)

# %%
# save all_res tensor 
torch.save(all_res, os.path.join(get_path(["uce_paths", "tmp_feature_path"]),"all_res_IG_20steps.torch") )
