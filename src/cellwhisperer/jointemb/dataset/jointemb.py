from torch.utils.data import Dataset, DataLoader
import anndata

import torch
import random
import logging

import torch
import lightning as pl

from transformers import AutoTokenizer
from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor

from cellwhisperer.config import get_path, model_path_from_name

from typing import Optional, Dict


class JointEmbedDataset(Dataset):
    """
    Dataset of dicts
    """

    def __init__(self, inputs, orig_ids=None, replicate_inputs: Dict = {}):
        self.inputs = inputs
        self.orig_ids = orig_ids
        self.replicate_inputs = replicate_inputs

    def __len__(self):
        return len(next(iter(self.inputs.values())))

    def __getitem__(self, idx):
        sample = {key: val[idx] for key, val in self.inputs.items()}
        return sample

    def set_epoch(self, epoch: int):
        """
        Update the dataset to "load" the replicate for a specific epoch
        """
        for key, feature_data in self.replicate_inputs.items():
            self.inputs[key] = feature_data[epoch % len(feature_data)]


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
            "model_max_length": 128  # 128 seems to be a decent fit (previously 100)
        },  # see https://github.com/epigen/cellwhisperer/issues/193
        min_genes=1,
        train_fraction=0.95,
    ):
        """

        Note: This is also used after training in `post_clip_processing` (val_dataloader)

        Args:
            tokenizer: name of the tokenizer to use. Must be a valid name for the AutoTokenizer.from_pretrained() function.
            transcriptome_processor: name of the transcriptome processor to use. Must be a valid name for the GeneformerTranscriptomeProcessor class.
            dataset_name: name of the dataset to use. Must be a valid name for the get_path() function.
            batch_size: batch size to use for training and validation
            nproc: number of processes to use for transcriptome processing
            min_genes: minimum number of genes to use for a sample. A larger value may increase the dataset quality. Choose a value > 0 to prevent NaNs, which can occur when the number of genes is 0
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
            hash="_".join(
                [
                    self.transcriptome_processor,
                    tokenizer,
                    str(self.min_genes),
                ]
            ),
        )
        self.train_fraction = train_fraction
        self.transcriptome_processor_kwargs = transcriptome_processor_kwargs.copy()
        self.tokenizer_kwargs = tokenizer_kwargs.copy()

    def prepare_data(self, force_prepare=False):
        # check whether data has already been prepared
        if self.processed_path.exists() and not force_prepare:
            logging.info("data already prepared")
            # return
        logging.info("preparing data...")

        processor = TranscriptomeTextDualEncoderProcessor(
            self.transcriptome_processor,
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
        # Add weights tensors (if available)
        for modality_weights_key in ["transcriptome_weights", "annotation_weights"]:
            if modality_weights_key in adata.obs:
                inputs[modality_weights_key] = torch.from_numpy(
                    adata.obs[modality_weights_key]
                )

        # Take the length from the first one and apply it to all of them
        # This is not required per se (the dimensionalities could also vary from epoch to epoch), however, it feels cleaner to remain dimensionalities across epochs.
        # We could also stack them into a single tensor, but I see little benefit at the moment
        # Note that the first one is computed twice (redundantly)
        max_length = inputs["input_ids"].shape[1]

        replicate_inputs = {
            "input_ids": [],
            "attention_masks": [],
        }

        # TODO generalize this later towards adata.layers for single-cell replicates
        if "natural_language_annotation_replicates" in adata.uns:
            replicate_df = adata.uns["natural_language_annotation_replicates"]
            logging.info(f"Loading {len(replicate_df.columns)} replicate annotations")
            for col_name in replicate_df:
                replicate_annotations = replicate_df[col_name]
                replicate_input = processor(
                    text=replicate_annotations.to_list(),
                    return_tensors="pt",
                    padding="max_length",  # enforces fixed size (https://huggingface.co/docs/transformers/en/pad_truncation)
                    max_length=max_length,
                )
                for key, feature_data in replicate_input:
                    replicate_inputs[key].append(feature_data)

        # Take the length from the first one and apply it to all of them
        # This is not required per se (the dimensionalities could also vary from epoch to epoch), however, it feels cleaner to remain dimensionalities across epochs.
        # We could also stack them into a single tensor, but I see little benefit at the moment
        # Note that the first one is computed twice (redundantly)
        max_length = inputs["input_ids"].shape[1]

        replicate_inputs = {
            "input_ids": [],
            "attention_masks": [],
        }

        # TODO generalize this later towards adata.layers for single-cell replicates
        if "natural_language_annotation_replicates" in adata.uns:
            replicate_df = adata.uns["natural_language_annotation_replicates"]
            logging.info(f"Loading {len(replicate_df.columns)} replicate annotations")
            for col_name in replicate_df:
                replicate_annotations = replicate_df[col_name]
                replicate_input = processor(
                    text=replicate_annotations.to_list(),
                    return_tensors="pt",
                    padding="max_length",  # enforces fixed size (https://huggingface.co/docs/transformers/en/pad_truncation)
                    max_length=max_length,
                )
                for key, feature_data in replicate_input:
                    replicate_inputs[key].append(feature_data)

        # Filter for empty inputs
        if self.transcriptome_processor == "geneformer":
            n_genes_filter = inputs["expression_token_lengths"] >= self.min_genes
        elif self.transcriptome_processor == "scgpt":
            n_genes_filter = (inputs["expression_key_padding_mask"] == False).sum(
                dim=1
            ) >= self.min_genes
        else:
            raise ValueError(
                "Transcriptome processor {self.transcriptome_processor} not supported"
            )

        if sum(n_genes_filter) == len(n_genes_filter):
            logging.info(
                f"No samples were filtered out (All cells had >= {self.min_genes} genes)"
            )
        else:
            inputs = {key: val[n_genes_filter] for key, val in inputs.items()}
            inputs["orig_ids"] = adata.obs.index[n_genes_filter]
            if len(replicate_inputs["input_ids"]) > 0:
                raise NotImplementedError(
                    "would also need to filter the replicates fields"
                )

            logging.warning(
                f"Filtered for {sum(n_genes_filter)} of {len(n_genes_filter)} samples with >={self.min_genes} genes."
            )

        # save the inputs dict to a file using torch
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            (inputs, replicate_inputs),
            self.processed_path,
        )

    def setup(self, stage=None):
        (inputs, replicate_inputs) = torch.load(self.processed_path)
        # Assuming you want to split the data into train and val for simplicity
        train_size = int(self.train_fraction * len(inputs["input_ids"]))
        # randomly sample train_size indices for train and use the rest for val
        # fix the seed
        random.seed(42)
        total_ids = list(range(len(inputs["input_ids"])))
        train_ids = random.sample(total_ids, train_size)
        val_ids = sorted(list(set(total_ids) - set(train_ids)))

        self.train_dataset = JointEmbedDataset(
            {
                key: value[train_ids]
                for key, value in inputs.items()
                if key != "orig_ids"
            },
            orig_ids=inputs["orig_ids"][train_ids],
            replicate_inputs={
                key: [value[i][train_ids] for i in train_ids]
                for key, value in replicate_inputs.items()
            },
        )
        self.val_dataset = JointEmbedDataset(
            {key: value[val_ids] for key, value in inputs.items() if key != "orig_ids"},
            orig_ids=inputs["orig_ids"][val_ids],
        )

    def train_dataloader(self):
        self.train_dataset.set_epoch(self.trainer.current_epoch)
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
            drop_last=False,
            shuffle=False,
        )
