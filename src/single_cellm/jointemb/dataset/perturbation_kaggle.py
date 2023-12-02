# from pytorch_metric_learning import samplers
from pathlib import Path

# import torchvision.transforms as transforms
# import imageio
import torch

# from pytorch_metric_learning.utils import common_functions as c_f
# from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import lightning as pl

from transformers import AutoTokenizer
from single_cellm.config import get_path
import pandas as pd
import numpy as np


NUM_THREADS = 20


class KaggleDEGDataSet(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs["control_features"]["input_ids"])

    def __getitem__(self, idx):
        sample = {
            "control_features": {
                key: val[idx] for key, val in self.inputs["control_features"].items()
            },
            "perturbation_features": {
                key: val[idx]
                for key, val in self.inputs["perturbation_features"].items()
            },
            "labels": torch.from_numpy(self.inputs["labels"].iloc[idx].values)
            if "labels" in self.inputs
            else None,
        }
        return sample


def collate(batch):
    return {
        "control_features": {
            key: torch.stack([b["control_features"][key] for b in batch])
            for key in batch[0]["control_features"].keys()
        },
        "perturbation_features": {
            key: torch.stack([b["perturbation_features"][key] for b in batch])
            for key in batch[0]["perturbation_features"].keys()
        },
        "labels": torch.stack([b["labels"] for b in batch])
        if batch[0]["labels"] is not None
        else None,
    }


class KaggleDEGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        deg_df,
        test_map,
        embedding_sentence,
        tokenizer,
        n_folds=3,
        kth_fold=0,
        batch_size=32,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.deg_df = deg_df
        self.test_map = test_map
        self.embedding_sentence = embedding_sentence
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.n_folds = n_folds
        self.kth_fold = kth_fold
        self.processed_path = get_path(
            ["paths", "datamodule_prepared_path"],
            dataset="perturbation_kaggle_deg",
            transcriptome_processor="",
            tokenizer=tokenizer,
        )

    def tokenize_df(self, df):
        perturbation_sentences = df.apply(
            lambda row: self.embedding_sentence.format(
                cell_type=row["cell_type"],
                sm_name=row["sm_name"],
                sm_lincs_id=row["sm_lincs_id"],
            ),
            axis=1,
        )

        control_sentences = df.apply(
            lambda row: self.embedding_sentence.format(
                cell_type=row["cell_type"],
                sm_name="Dimethyl Sulfoxide",
                sm_lincs_id="LSM-36361",
            ),
            axis=1,
        )

        inputs_perturbed = self.tokenizer(
            text=list(perturbation_sentences),
            return_tensors="pt",
            padding=True,
        )

        inputs_control = self.tokenizer(
            text=list(control_sentences),
            return_tensors="pt",
            padding=True,
        )

        return inputs_perturbed, inputs_control

    def prepare_data(self):
        if self.processed_path.exists():
            print("data already prepared, not skipping though, just to make sure")
            return
        print("preparing data...")

        inputs_perturbed, inputs_control = self.tokenize_df(self.deg_df)
        inputs_test_perturbed, inputs_test_control = self.tokenize_df(self.test_map)

        # save the inputs dict to a file using torch
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            (
                inputs_perturbed,
                inputs_control,
                inputs_test_perturbed,
                inputs_test_control,
            ),
            self.processed_path,
        )

    def setup(self, stage=None):
        # define train test dataset
        (
            inputs_perturbed,
            inputs_control,
            inputs_test_perturbed,
            inputs_test_control,
        ) = torch.load(self.processed_path)

        df = self.deg_df
        b_cells = df.index[
            (df["cell_type"] == "B cells") & (~df["control"])
        ]  # use the positive controls for training
        myeloid_cells = df.index[
            (df["cell_type"] == "Myeloid cells") & (~df["control"])
        ]
        # all_others =  df.index.subtract(b_cells).subtract(myeloid_cells)

        l = len(b_cells)
        assert len(b_cells) == len(myeloid_cells)

        n = self.n_folds
        val_splits = [
            pd.Index.union(
                b_cells[int(i * l / n) : int((i + 1) * l / n)],
                myeloid_cells[int(i * l / n) : int((i + 1) * l / n)],
            )
            for i in range(n)
        ]

        train_splits = [df.index.difference(val) for val in val_splits]

        train_ids = train_splits[self.kth_fold]
        val_ids = val_splits[self.kth_fold]

        self.train_dataset = KaggleDEGDataSet(
            {
                "control_features": {
                    key: val[train_ids] for key, val in inputs_control.items()
                },
                "perturbation_features": {
                    key: val[train_ids] for key, val in inputs_perturbed.items()
                },
                "labels": df.loc[train_ids].select_dtypes(include=[np.number]),
            }
        )
        self.val_dataset = KaggleDEGDataSet(
            {
                "control_features": {
                    key: val[val_ids] for key, val in inputs_perturbed.items()
                },
                "perturbation_features": {
                    key: val[val_ids] for key, val in inputs_control.items()
                },
                "labels": df.loc[val_ids].select_dtypes(include=[np.number]),
            }
        )

        self.test_dataset = KaggleDEGDataSet(
            {
                "control_features": inputs_test_control,
                "perturbation_features": inputs_test_perturbed,
            }
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, collate_fn=collate
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, collate_fn=collate
        )
