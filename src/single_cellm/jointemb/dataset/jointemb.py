from torch.utils.data import Dataset, DataLoader
import anndata

from pathlib import Path

import torch
import random
import logging

import torch
import lightning as pl

from transformers import AutoTokenizer
from single_cellm.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from single_cellm.jointemb.scgpt_model import ScGPTTranscriptomeProcessor
from single_cellm.jointemb.processing import TranscriptomeTextDualEncoderProcessor

from single_cellm.config import get_path, model_path_from_name
import subprocess


class JointEmbedDataset(Dataset):
    """
    Dataset of dicts
    """

    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(next(iter(self.inputs.values())))

    def __getitem__(self, idx):
        sample = {key: val[idx] for key, val in self.inputs.items()}
        return sample


class JointEmbedDataModule(pl.LightningDataModule):
    """
    Generates training/validation datasets containing matching transcriptome-annotation pairs.

    Data is loaded from AnnData objects (.h5ad). For processing details refer to TranscriptomeTextDualEncoderProcessor
    """

    def __init__(
        self,
        tokenizer="biogpt",
        transcriptome_processor="geneformer",
        dataset_name="daniel",
        batch_size=32,
        nproc=8,
        transcriptome_processor_kwargs={},
        tokenizer_kwargs={
            "model_max_length": 100  # 100 seems to be a decent fit, which cuts very few inputs (both for biogpt and biobert)
        },  # see https://github.com/epigen/single-cellm/issues/193
        min_genes=200,
    ):
        """
        Args:
            tokenizer: name of the tokenizer to use. Must be a valid name for the AutoTokenizer.from_pretrained() function.
            transcriptome_processor: name of the transcriptome processor to use. Must be a valid name for the GeneformerTranscriptomeProcessor class.
            dataset_name: name of the dataset to use. Must be a valid name for the get_path() function.
            batch_size: batch size to use for training and validation
            nproc: number of processes to use for transcriptome processing
            min_genes: minimum number of genes to use for a sample. This increases the dataset quality, but also prevents NaNs, which can occur when the number of genes is 0
        """
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.tokenizer = model_path_from_name(tokenizer)
        self.transcriptome_processor = transcriptome_processor
        self.nproc = nproc
        self.min_genes = min_genes
        self.processed_path = get_path(
            ["paths", "datamodule_prepared_path"],
            dataset=self.dataset_name,
            transcriptome_processor=self.transcriptome_processor,
            tokenizer=tokenizer,
        )
        self.transcriptome_processor_kwargs = transcriptome_processor_kwargs.copy()
        self.tokenizer_kwargs = tokenizer_kwargs.copy()

    def prepare_data(self):
        # check whether data has already been prepared
        if self.processed_path.exists():
            logging.info("data already prepared")
            return
        logging.info("preparing data...")

        if self.transcriptome_processor == "geneformer":
            transcriptome_processor = GeneformerTranscriptomeProcessor(
                nproc=self.nproc,
                emb_label="natural_language_annotation",  # config["anndata_label_name"]
                **self.transcriptome_processor_kwargs,
            )
        elif self.transcriptome_processor == "scgpt":
            transcriptome_processor = ScGPTTranscriptomeProcessor(
                nproc=self.nproc,
                **self.transcriptome_processor_kwargs,
            )

        else:
            raise ValueError("transcriptome_processor not recognized")

        processor = TranscriptomeTextDualEncoderProcessor(
            transcriptome_processor,
            AutoTokenizer.from_pretrained(self.tokenizer, **self.tokenizer_kwargs),
        )
        adata = anndata.read_h5ad(
            (get_path(["paths", "full_dataset"], dataset=self.dataset_name))
        )

        inputs = processor(
            text=list(adata.obs["natural_language_annotation"]),
            transcriptomes=adata,
            return_tensors="pt",
            padding=True,
        )
        # Filter for empty inputs
        if self.transcriptome_processor == "geneformer":
            n_genes_filter = inputs["expression_token_lengths"] > self.min_genes
        elif self.transcriptome_processor == "scgpt":
            n_genes_filter = (inputs["expression_key_padding_mask"] == False).sum(
                dim=1
            ) > self.min_genes
        else:
            raise ValueError(
                "Transcriptome processor {self.transcriptome_processor} not supported"
            )

        inputs = {key: val[n_genes_filter] for key, val in inputs.items()}
        logging.info(
            f"Filtered for {sum(n_genes_filter)} of {len(n_genes_filter)} samples with >{self.min_genes} genes."
        )

        # save the inputs dict to a file using torch
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            inputs,
            self.processed_path,
        )

    def setup(self, stage=None):
        inputs = torch.load(self.processed_path)
        # Assuming you want to split the data into train and val for simplicity
        train_size = int(0.8 * len(inputs["input_ids"]))
        # randomly sample train_size indices for train and use the rest for val
        # fix the seed
        random.seed(42)
        total_ids = list(range(len(inputs["input_ids"])))
        train_ids = random.sample(total_ids, train_size)
        val_ids = sorted(list(set(total_ids) - set(train_ids)))

        self.train_dataset = JointEmbedDataset(
            {key: val[train_ids] for key, val in inputs.items()}
        )
        self.val_dataset = JointEmbedDataset(
            {key: val[val_ids] for key, val in inputs.items()}
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.nproc,
            drop_last=True,  # drop last batch to avoid batch_size of 1, which fails due to batch-norm
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.nproc,
            drop_last=True,  # drop last batch to avoid batch_size of 1, which fails due to batch-norm
        )
