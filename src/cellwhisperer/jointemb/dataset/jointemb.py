from torch.utils.data import Dataset, DataLoader
import anndata
from collections import defaultdict
from itertools import zip_longest

import torch
import random
import logging
import pandas as pd

import torch
from torch.utils.data import ConcatDataset
import lightning as pl

from transformers import AutoTokenizer
from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor

from cellwhisperer.config import get_path, model_path_from_name

from typing import Optional, Dict, Union

logger = logging.getLogger(__name__)


class JointEmbedDataset(Dataset):
    """
    Dataset of dicts
    """

    def __init__(self, inputs, orig_ids=None, replicate_inputs: Dict = {}):
        self.inputs = inputs
        self.orig_ids = orig_ids
        self.orig_inputs = inputs  # save for being able to switch around replicates
        self.replicate_inputs = replicate_inputs

    def __len__(self):
        return len(next(iter(self.inputs.values())))

    def __getitem__(self, idx):
        sample = {key: val[idx] for key, val in self.inputs.items()}
        return sample

    def set_epoch(self, epoch: int):
        """
        Update the dataset to "load" the replicate for a specific epoch.

        Treat the original input as one additional replicate
        """
        # get length of replicate dict elements
        try:
            num_replicates = len(next(iter(self.replicate_inputs.values())))
        except StopIteration:
            logging.info("No replicates found. Fall back to 0")
            num_replicates = 0

        # add 1 because of the original input
        replicate_i = epoch % (num_replicates + 1)

        if replicate_i == 0:
            self.inputs = self.orig_inputs
        else:
            logging.info(f"Selecting replicate #{replicate_i} of {num_replicates}")
            for key, feature_data in self.replicate_inputs.items():
                self.inputs[key] = feature_data[replicate_i - 1]


class JointEmbedDataModule(pl.LightningDataModule):
    """
    Generates training/validation datasets containing matching transcriptome-annotation pairs.

    Data is loaded from AnnData objects (.h5ad). For processing details refer to TranscriptomeTextDualEncoderProcessor
    """

    def __init__(
        self,
        tokenizer="bert",
        transcriptome_processor="geneformer",
        dataset_names="human_disease",
        batch_size=32,
        nproc=8,
        transcriptome_processor_kwargs={},
        tokenizer_kwargs={
            "model_max_length": 128  # 128 seems to be a decent fit (previously 100)
        },  # see https://github.com/epigen/cellwhisperer/issues/193
        min_genes=100,
        train_fraction: Union[str, float] = 0.95,
        use_replicates: bool = True,
        include_labels: Optional[str] = None,
    ):
        """

        Note: This is also used after training in `cellxgene_preprocessing` (val_dataloader)

        Args:
            tokenizer: name of the tokenizer to use. Must be a valid name for the AutoTokenizer.from_pretrained() function.
            transcriptome_processor: name of the transcriptome processor to use. Must be a valid name for the GeneformerTranscriptomeProcessor class.
            dataset_names: Comma-separated list of names of the datasets to use. Must be valid name for the get_path() function.
            batch_size: batch size to use for training and validation
            nproc: number of processes to use for transcriptome processing
            min_genes: minimum number of genes to use for a sample. A larger value may increase the dataset quality. Choose a value > 0 to prevent NaNs, which can occur when the number of genes is 0
            train_fraction: fraction of the data to use for training. The rest will be used for validation.
            use_replicates: whether to use replicates for the transcriptome and annotation data.
            include_labels: name of the column in the AnnData object to use as labels. If None, no labels will be included
        """
        super().__init__()
        self.batch_size = batch_size
        self.dataset_names = dataset_names.split(",")
        self.tokenizer = model_path_from_name(tokenizer)
        self.transcriptome_processor = transcriptome_processor
        self.nproc = nproc
        self.min_genes = min_genes
        self.train_fraction = train_fraction
        self.transcriptome_processor_kwargs = transcriptome_processor_kwargs.copy()
        self.tokenizer_kwargs = tokenizer_kwargs.copy()
        self.use_replicates = use_replicates
        self.include_labels = include_labels

    def _processed_path(self, dataset_name):
        return get_path(
            ["paths", "datamodule_prepared_path"],
            dataset=dataset_name,
            hash="_".join(
                [
                    self.transcriptome_processor,
                    "" if not self.tokenizer else self.tokenizer.replace("/", "__"),
                    str(self.min_genes),
                    str(self.use_replicates),
                    self.include_labels or "",
                ]
            ),
        )

    def prepare_data(self, force_prepare=False):
        # Generate all datasets
        for dataset_name in self.dataset_names:
            self.prepare_data_single(dataset_name, force_prepare)

    def prepare_data_single(self, dataset_name, force_prepare):
        processed_path = self._processed_path(dataset_name)

        # check whether data has already been prepared
        if processed_path.exists() and not force_prepare:
            logger.info("data already prepared")
            return
        logger.info("preparing data...")

        # Load data and processor
        processor = TranscriptomeTextDualEncoderProcessor(
            self.transcriptome_processor,
            (
                AutoTokenizer.from_pretrained(self.tokenizer, **self.tokenizer_kwargs)
                if self.tokenizer
                else None
            ),
        )
        adata = anndata.read_h5ad(
            (get_path(["paths", "full_dataset"], dataset=dataset_name))
        )

        # Fixed size embedding (https://huggingface.co/docs/transformers/en/pad_truncation), as we combine multiple datasets
        inputs = processor(
            text=(
                list(adata.obs["natural_language_annotation"])
                if self.tokenizer
                else None
            ),
            transcriptomes=adata,
            return_tensors="pt",
            padding="max_length",
        )

        # Add weights tensors (if available)
        for modality_weights_key in ["transcriptome_weights", "annotation_weights"]:
            if modality_weights_key in adata.obs:
                inputs[modality_weights_key] = torch.from_numpy(
                    adata.obs[modality_weights_key].values
                )
            else:
                # add 1 if no weights are available
                inputs[modality_weights_key] = torch.ones(len(adata.obs))

        logging.info("Preparing replicates")
        if self.use_replicates:
            replicate_inputs = self._prepare_replicate_inputs(inputs, adata, processor)
        else:
            replicate_inputs = {}

        if self.include_labels is not None:
            if adata.obs[self.include_labels].dtype.name != "category":
                adata.obs[self.include_labels] = pd.Categorical(
                    adata.obs[self.include_labels]
                )
                # NOTE maybe save the categorical as well for later?

            # Add labels to the inputs.
            inputs["labels"] = torch.tensor(
                adata.obs[self.include_labels].astype("category").cat.codes.values,
                dtype=torch.long,
            )

        # Filter for empty inputs (NOTE: empty inputs should be avoided)
        if self.transcriptome_processor == "geneformer":
            n_genes_filter = inputs["expression_token_lengths"] >= self.min_genes
        elif self.transcriptome_processor == "scgpt":
            n_genes_filter = (inputs["expression_key_padding_mask"] == False).sum(
                dim=1
            ) >= self.min_genes
        elif self.transcriptome_processor == "uce":
            n_genes_filter = (inputs["expression_key_padding_mask"] == False).sum(
                dim=1
            ) >= 1  # self.min_genes  # NOTE the mask cannot be used unfortunately for filtering
        else:
            raise ValueError(
                "Transcriptome processor {self.transcriptome_processor} not supported"
            )

        if sum(n_genes_filter) == len(n_genes_filter):
            logger.info(
                f"No samples were filtered out (All cells had >= {self.min_genes} genes)"
            )
            inputs["orig_ids"] = adata.obs.index[n_genes_filter]
        else:
            logger.warning(
                f"Filtering for {sum(n_genes_filter)} of {len(n_genes_filter)} samples with >={self.min_genes} genes."
            )
            inputs = {key: val[n_genes_filter] for key, val in inputs.items()}
            inputs["orig_ids"] = adata.obs.index[n_genes_filter]
            if len(replicate_inputs) > 0:
                for key, rep_value in replicate_inputs.items():
                    replicate_inputs[key] = [
                        value[n_genes_filter] for value in rep_value
                    ]
                    assert len(rep_value) > 0

        # save the inputs dict to a file using torch
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            (inputs, replicate_inputs),
            processed_path,
        )

    def _prepare_replicate_inputs(self, inputs, adata, processor):
        """
        Take the length from the first one and apply it to all of them
        This is not required per se (the dimensionalities could also vary from epoch to epoch), however, it feels cleaner to remain dimensionalities across epochs.
        We could also stack them into a single tensor, but I see little benefit at the moment
        Note that the first one is computed twice (redundantly)

        """

        replicate_inputs = defaultdict(list)

        # Annotation replicates
        if (
            "natural_language_annotation_replicates" in adata.obsm
            and adata.obsm["natural_language_annotation_replicates"].shape[1] > 0
        ):
            max_length = inputs["input_ids"].shape[
                1
            ]  # NOTE this is actually redundant as we anyways force it to 128
            replicate_df = adata.obsm["natural_language_annotation_replicates"]
            logger.info(f"Loading {len(replicate_df.columns)} replicate annotations")
            for col_name in replicate_df:
                replicate_annotations = replicate_df[col_name]
                replicate_input = processor(
                    text=replicate_annotations.to_list() if self.tokenizer else None,
                    return_tensors="pt",
                    padding="max_length",  # enforces fixed size (https://huggingface.co/docs/transformers/en/pad_truncation)
                    max_length=max_length,
                )
                for key, feature_data in replicate_input.items():
                    replicate_inputs[key].append(feature_data)

        # Transcriptome replicates
        replicate_layers = sorted(
            [
                key
                for key in adata.layers.keys()
                if "replicate" in key or "sampled_cell" in key
            ]
        )
        logger.info(f"Loading {len(replicate_layers)} replicate transcriptomes")
        X = adata.X
        for layer_name in replicate_layers:
            adata.X = adata.layers[layer_name]
            replicate_input = processor(
                transcriptomes=adata,
                return_tensors="pt",
            )
            for key, feature_data in replicate_input.items():
                replicate_inputs[key].append(feature_data)
        # restore adata.X to the original value
        adata.X = X

        return replicate_inputs

    def setup(self, stage=None):
        self.train_datasets = []
        self.val_datasets = []
        for dataset_name in self.dataset_names:
            (inputs, replicate_inputs) = torch.load(self._processed_path(dataset_name))

            if isinstance(self.train_fraction, (int, float)):
                # Assuming you want to split the data into train and val for simplicity
                train_size = int(self.train_fraction * len(inputs["input_ids"]))
                # randomly sample train_size indices for train and use the rest for val
                # fix the seed
                random.seed(42)
                total_ids = list(range(len(inputs["input_ids"])))
                train_ids = random.sample(total_ids, train_size)
                val_ids = sorted(list(set(total_ids) - set(train_ids)))
            elif isinstance(self.train_fraction, str):
                total_ids = list(range(len(inputs["input_ids"])))
                if dataset_name == self.train_fraction:
                    train_ids = total_ids
                    val_ids = []
                else:
                    train_ids = []
                    val_ids = total_ids
            else:
                raise ValueError("train_fraction must be either a float or string")

            self.train_datasets.append(
                JointEmbedDataset(
                    {
                        key: value[train_ids]
                        for key, value in inputs.items()
                        if key != "orig_ids"
                    },
                    orig_ids=inputs["orig_ids"][train_ids],
                    replicate_inputs={
                        key: [value[i][train_ids] for i in range(len(value))]
                        for key, value in replicate_inputs.items()
                        if len(value) > 0
                    },
                )
            )
            self.val_datasets.append(
                JointEmbedDataset(
                    {
                        key: value[val_ids]
                        for key, value in inputs.items()
                        if key != "orig_ids"
                    },
                    orig_ids=inputs["orig_ids"][val_ids],
                )
            )

    def train_dataloader(self):
        # Update the current epoch to sample the replicates
        if self.use_replicates:
            for dataset in self.train_datasets:
                dataset.set_epoch(self.trainer.current_epoch)

        return DataLoader(
            ConcatDataset(self.train_datasets),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.nproc,
            drop_last=True,  # drop last batch to avoid batch_size of 1, which fails due to batch-norm
        )

    def val_dataloader(self):
        """
        In principle, we could also return a list of DataLoaders, but it is currently incompatible with RetrievalScoreCalculator
        """
        return DataLoader(
            ConcatDataset(self.val_datasets),
            batch_size=self.batch_size,
            num_workers=self.nproc,
            drop_last=False,  # more accurate if we don't drop the last
            shuffle=False,
        )

    def test_dataloader(self):
        # Return the validation dataloader for testing
        return self.val_dataloader()
