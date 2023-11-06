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
from torch.utils.data import Dataset, DataLoader
import lightning as pl

from transformers import AutoTokenizer
from single_cellm.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from single_cellm.jointemb.processing import TranscriptomeTextDualEncoderProcessor
import subprocess

PROJECT_DIR = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=Path(__file__).parent
    )
    .decode("utf-8")
    .strip()
)
NUM_THREADS = 20


class JointEmbedDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        sample = {key: val[idx] for key, val in self.inputs.items()}
        return sample


class JointEmbedDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name="daniel", batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.processed_path = (
            PROJECT_DIR / f"results/{self.dataset_name}/lightning-processed/inputs.pt"
        )  # TODO this path needs to go to config (and/or provided as argument)

    def prepare_data(self):
        # TODO check whether data has already been prepared
        if self.processed_path.exists():
            print("data already prepared, skipping...")
            return
        print("preparing data...")

        tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
        transcriptome_processor = GeneformerTranscriptomeProcessor(
            nproc=NUM_THREADS,
            emb_label="natural_language_annotation",  # config["anndata_label_name"]
        )
        processor = TranscriptomeTextDualEncoderProcessor(
            transcriptome_processor, tokenizer
        )
        adata = anndata.read_h5ad(  # TODO provide path through function parameters and config.yaml :)
            PROJECT_DIR / f"results/{self.dataset_name}/full_data.h5ad"
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
