"""
Providing KOs captured within the DepMap project (https://depmap.org/portal/).

DepMap provides a list of 1856 genes that are thought to be essential across cancer cell lines, which can be downloaded from https://depmap.org/portal/download/all/?releasename=DepMap+Public+22Q4&filename=CRISPRInferredCommonEssentials.csv . There is also a list of non-essential controls (https://depmap.org/portal/download/all/?releasename=DepMap+Public+22Q4&filename=AchillesNonessentialControls.csv).
"""


from single_cellm.jointemb.dataset import JointEmbedDataset
from lightning import LightningDataModule
import logging
from single_cellm.jointemb.processing import TranscriptomeTextDualEncoderProcessor
from torch.utils.data import Dataset, DataLoader
from single_cellm.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from single_cellm.jointemb.scgpt_model import ScGPTTranscriptomeProcessor
from transformers import AutoTokenizer
import anndata
from single_cellm.config import get_path, model_path_from_name
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import torch

import pandas as pd
import re
from transformers import AutoTokenizer


class CancerGeneEssentialityDataModule(LightningDataModule):
    """
    Loads gene essentiality dataset as well as some transcriptomics data. Generates an in silico KO dataset for all the (non-)essential genes.


    Takes the first sample in `dataset_name` and populates it with the genes in `df_essentiality`, perform knockouts.
    """

    def __init__(
        self,
        tokenizer="microsoft/biogpt",
        transcriptome_processor="geneformer",
        dataset_name="daniel",
        batch_size=32,
        transcriptome_processor_kwargs={},
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name

        if dataset_name != "daniel":
            logging.warning(
                f"Dataset {dataset_name} might not be supported for CancerGeneEssentialityDataModule. Make sure its `var` object contains the `gene_name` field."
            )

        self.tokenizer = model_path_from_name(tokenizer)
        self.transcriptome_processor = transcriptome_processor
        self.processed_path = get_path(
            ["paths", "datamodule_prepared_path"],
            dataset="cancer_gene_essentiality",
            transcriptome_processor=transcriptome_processor,
            tokenizer=tokenizer,
        )
        self.transcriptome_processor_kwargs = transcriptome_processor_kwargs

    def prepare_data(self):
        """
        Dataset-specific preparation
        """
        if self.processed_path.exists():
            logging.info("data already prepared...")
            return
        essential = pd.read_csv(
            get_path(["paths", "cancer_gene_essentiality", "essential_genes"])
        )
        essential["essential"] = True
        nonessential = pd.read_csv(
            get_path(["paths", "cancer_gene_essentiality", "nonessential_genes"])
        )
        nonessential["essential"] = False

        df_essent = pd.concat(
            [essential.rename(columns={"Essentials": "Gene"}), nonessential]
        )
        df_essent["quant"] = df_essent["Gene"].apply(
            lambda v: int(re.search(r"[0-9]+\)$", v).group()[:-2])
        )
        df_essent["Gene"] = df_essent["Gene"].apply(
            lambda v: re.search(r"^.+ ", v).group()[:-1]
        )

        self.processor = TranscriptomeTextDualEncoderProcessor(
            self.transcriptome_processor,
            AutoTokenizer.from_pretrained(self.tokenizer),
            **self.transcriptome_processor_kwargs,
        )
        # read out the first transcriptome from our dataset
        adata = anndata.read_h5ad(
            (get_path(["paths", "full_dataset"], dataset=self.dataset_name))
        )[:1]

        # for each gene in the provided lists, generate an artificial knockout
        gene_masks = df_essent["Gene"].apply(lambda x: (adata.var.gene_name == x))

        if issparse(adata.X):
            # Convert the sparse matrix to a NumPy ndarray
            X = adata.X.toarray()
        else:
            # It's already a NumPy ndarray, so we can use it directly
            X = np.array(adata.X)
        # boost the single sample (first dimension) to len(df_essent) samples
        X = np.repeat(X, len(df_essent), axis=0)
        # introduce the knockout
        X[gene_masks.to_numpy()] = 0

        # use the first entry in adata.obs and populate it len(df_essentiality) times
        obs = pd.DataFrame([adata.obs.iloc[0]] * len(df_essent))
        obs["natural_language_annotation"] = df_essent["Gene"].values  # unused
        obs.index = obs.apply(
            lambda row: f"{row.name}_KO_{row.natural_language_annotation}", axis=1
        )

        adata = anndata.AnnData(
            X=X,
            var=pd.DataFrame(adata.var),
            obs=obs,
        )

        data = self.processor(
            transcriptomes=adata,
            return_tensors="pt",
            padding=True,
        )
        data["labels"] = df_essent["essential"].values
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            data,
            self.processed_path,
        )

    def setup(self, stage=None):
        self.data = torch.load(self.processed_path)

    def predict_dataloader(self):
        return DataLoader(JointEmbedDataset(self.data), batch_size=self.batch_size)
