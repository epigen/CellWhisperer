"""
Providing KOs captured within the DepMap project (https://depmap.org/portal/).

DepMap provides a list of 1856 genes that are thought to be essential across cancer cell lines, which can be downloaded from https://depmap.org/portal/download/all/?releasename=DepMap+Public+22Q4&filename=CRISPRInferredCommonEssentials.csv . There is also a list of non-essential controls (https://depmap.org/portal/download/all/?releasename=DepMap+Public+22Q4&filename=AchillesNonessentialControls.csv).

Cancer cell line expressions were downloaded from URL: https://depmap.org/portal/download/all filename: CCLE_RNAseq_genes_counts_20180929.gct.gz
pandas.read_csv(PATH_TO_GCT_FILE, sep='\t',skiprows=2)

"""


from typing import Optional
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
        dataset_name="ccle",
        batch_size=32,
        transcriptome_processor_kwargs={},
        num_transcriptomes: Optional[int] = 2,
    ):
        """
        Args:
            tokenizer: name of the tokenizer to use. Must be a valid name for the AutoTokenizer.from_pretrained() function.
            transcriptome_processor: name of the transcriptome processor to use. Must be a valid name for the GeneformerTranscriptomeProcessor class.
            dataset_name: name of the dataset to use. Must be a valid name for the get_path() function.
            batch_size: batch size to use for training and validation
            transcriptome_processor_kwargs: kwargs to pass to the transcriptome processor
            num_transcriptomes: number of transcriptomes to use (pass None to use all). Be careful as large values might not fit in memory.
        """
        super().__init__()
        self.num_transcriptomes = num_transcriptomes
        self.batch_size = batch_size
        self.dataset_name = dataset_name

        if dataset_name != "ccle":
            logging.warning(
                f"Dataset {dataset_name} might not be supported for CancerGeneEssentialityDataModule. Make sure its `var` object contains the `gene_name` field. e.g. `daniel` should work"
            )

        self.tokenizer = model_path_from_name(tokenizer)
        self.transcriptome_processor = transcriptome_processor
        self.processed_path = get_path(
            ["paths", "datamodule_prepared_path"],
            dataset=f"cancer_gene_essentiality_{dataset_name}",
            hash="_".join(
                [
                    transcriptome_processor,
                    tokenizer,
                    str(num_transcriptomes),
                ]
            ),
        )
        self.transcriptome_processor_kwargs = transcriptome_processor_kwargs

    def prepare_adata(self, df_essent):
        if self.dataset_name == "ccle":
            ccle_df = pd.read_csv(
                get_path(
                    ["paths", "cancer_gene_essentiality", "ccle_expression"],
                    compression="gzip",
                ),
                sep="\t",
                skiprows=2,
                index_col=[0, 1],
            )
            ccle_df.index.names = ["ensembl_id", "gene_name"]
            # sample `self.num_transcriptomes` columns
            if self.num_transcriptomes is not None:
                assert self.num_transcriptomes <= ccle_df.shape[1]
                ccle_df = ccle_df.sample(
                    self.num_transcriptomes, axis=1, random_state=42
                )

            # Extract the gene expression matrix and cell line names
            X = ccle_df.values.T  # Assuming the first two columns are gene identifiers
            cell_lines = ccle_df.columns

            # Create a DataFrame for the cell line metadata
            obs = pd.DataFrame(cell_lines, columns=["cell_line"])

            # Repeat the metadata and the expression matrix for each gene knockout
            obs = pd.concat([obs] * len(df_essent), ignore_index=True)
            obs["essential"] = np.repeat(df_essent["essential"].values, len(cell_lines))
            X = np.tile(X, (len(df_essent), 1))

            # Create gene masks and apply knockouts
            for i, gene in enumerate(df_essent["Gene"]):
                if gene == "":  # control gene
                    continue
                gene_mask = ccle_df.index.get_level_values("gene_name") == gene
                X[i * len(cell_lines) : (i + 1) * len(cell_lines), gene_mask] = 0

            # Update the obs DataFrame with knockout information
            obs["gene_ko"] = np.repeat(df_essent["Gene"].values, len(cell_lines))
            obs["natural_language_annotation"] = obs.apply(
                lambda row: f"{row.cell_line}_KO_{row.gene_ko}",
                axis=1,
            )
            obs.set_index("natural_language_annotation", inplace=True, drop=False)

            # Create the var DataFrame with gene information (e.g. Description and ens)
            var = ccle_df.index.to_frame().set_index("gene_name", drop=False)

        else:  # e.g. self.dataset_name == "daniel" NOTE: this is deprecated. use ccle
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
            var = pd.DataFrame(adata.var)

        adata = anndata.AnnData(
            X=X,
            var=var,
            obs=obs,
        )
        return adata

    def prepare_data(self):
        """
        Dataset-specific preparation
        """
        if self.processed_path.exists():
            logging.info("data already prepared...")
            # return  # TODO enable (e.g. during sweep)
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
        # df_essent["quant"] = df_essent["Gene"].apply(
        #     lambda v: int(re.search(r"[0-9]+\)$", v).group()[:-2])
        # )
        df_essent["Gene"] = df_essent["Gene"].apply(
            lambda v: re.search(r"^.+ ", v).group()[:-1]
        )
        # Add a control_gene
        df_essent = pd.concat(
            [
                df_essent,
                pd.DataFrame({"Gene": [""], "essential": [False]}),  # "quant": [-1],
            ]
        )

        self.processor = TranscriptomeTextDualEncoderProcessor(
            self.transcriptome_processor,
            AutoTokenizer.from_pretrained(self.tokenizer),
            **self.transcriptome_processor_kwargs,
        )
        adata = self.prepare_adata(df_essent)

        data = self.processor(
            transcriptomes=adata,
            return_tensors="pt",
            padding=True,
        )

        data["essential"] = adata.obs["essential"].values

        # convert gene_ko and cell_line to discrete integers using torch
        data["gene_ko"] = np.unique(adata.obs["gene_ko"], return_inverse=True)[1]
        # Set the value for the control (None) to -1
        data["gene_ko"] = np.where(adata.obs["gene_ko"] == "", -1, data["gene_ko"])

        cell_lines, data["cell_line"] = np.unique(
            adata.obs["cell_line"], return_inverse=True
        )
        data["orig_ids"] = adata.obs.index.values

        # Remove all entries that are replicates of the control ones
        transcriptome_key = (
            "expression_expr" if "expression_expr" in data else "expression_tokens"
        )
        drop_mask = torch.zeros(*data["gene_ko"].shape, dtype=torch.bool)

        for i, _ in enumerate(cell_lines):
            control_mask = torch.from_numpy(
                (data["gene_ko"] == -1) & (data["cell_line"] == i)
            )
            control_transcriptome = data[transcriptome_key][control_mask]

            # Compare with all non-control transcriptomes
            duplicate_mask = torch.all(
                data[transcriptome_key] == control_transcriptome, dim=1
            )
            # only delete from the same cell_line
            duplicate_mask = duplicate_mask & torch.from_numpy(data["cell_line"] == i)

            # don't delete the control_transcriptome
            duplicate_mask = duplicate_mask & (~control_mask)
            drop_mask = drop_mask | duplicate_mask

        # filter all datas using the drop_mask
        data = {key: value[~drop_mask] for key, value in data.items()}

        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            data,
            self.processed_path,
        )

    def setup(self, stage=None):
        self.data = torch.load(self.processed_path)
        self.orig_ids = self.data.pop("orig_ids")

    def predict_dataloader(self):
        return DataLoader(
            JointEmbedDataset(self.data, orig_ids=self.orig_ids),
            batch_size=self.batch_size,
        )
