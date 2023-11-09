from torch.utils.data import Dataset, DataLoader
import anndata

# from pytorch_metric_learning import samplers
from pathlib import Path

# import torchvision.transforms as transforms
# import imageio
import torch
import random

# from pytorch_metric_learning.utils import common_functions as c_f
# from PIL import Image

import torch
import lightning as pl

from transformers import AutoTokenizer
from single_cellm.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from single_cellm.jointemb.processing import TranscriptomeTextDualEncoderProcessor

from single_cellm.config import get_path
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
        tokenizer="microsoft/biogpt",
        transcriptome_processor="geneformer",
        dataset_name="daniel",
        batch_size=32,
        nproc=8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.transcriptome_processor = transcriptome_processor
        self.nproc = nproc
        self.processed_path = get_path(
            ["paths", "datamodule_prepared_path"], dataset=self.dataset_name
        )

    def prepare_data(self):
        # check whether data has already been prepared
        if self.processed_path.exists():
            print(
                "data already prepared, repreparing anyways to make sure that config did not change..."
            )
            # return
        print("preparing data...")

        if self.transcriptome_processor == "geneformer":
            transcriptome_processor = GeneformerTranscriptomeProcessor(
                nproc=self.nproc,
                emb_label="natural_language_annotation",  # config["anndata_label_name"]
            )
        else:
            raise ValueError("transcriptome_processor not recognized")

        processor = TranscriptomeTextDualEncoderProcessor(
            transcriptome_processor, AutoTokenizer.from_pretrained(self.tokenizer)
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
        train_ids = random.sample(range(len(inputs["input_ids"])), train_size)
        val_ids = [i for i in range(len(inputs["input_ids"])) if i not in train_ids]

        self.train_dataset = JointEmbedDataset(
            {key: val[train_ids] for key, val in inputs.items()}
        )
        self.val_dataset = JointEmbedDataset(
            {key: val[val_ids] for key, val in inputs.items()}
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
