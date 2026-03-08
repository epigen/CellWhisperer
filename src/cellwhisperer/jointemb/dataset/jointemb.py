from torch.utils.data import Dataset, DataLoader
import anndata
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from tqdm import tqdm
import glob

import torch
import random
import logging
import pandas as pd

import torch
from torch.utils.data import ConcatDataset
from cellwhisperer.jointemb.conch_text_processor import ConchTextProcessor
import lightning as pl

from transformers import AutoTokenizer
from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor

from cellwhisperer.config import get_path, model_path_from_name

from typing import Optional, Dict, Union, List, Any

logger = logging.getLogger(__name__)


def multi_modal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for handling heterogeneous multi-modal data.

    This function handles batches where some samples may have text data,
    some may have image data, and some may have transcriptome data.
    It creates zero-filled tensors for missing modalities and generates
    modality masks to indicate which modalities are present.

    Args:
        batch: List of dictionaries containing sample data

    Returns:
        Dictionary containing:
        - Collated data for each modality (with zeros for missing data)
        - Modality masks: text_batch_mask, image_batch_mask, transcriptome_batch_mask
    """
    if not batch:
        return {}

    batch_size = len(batch)
    collated_batch = {}

    # Define modality keys for each type
    text_keys = ["input_ids", "attention_mask", "token_type_ids"]
    image_keys = ["patches_ctx", "patches_cell"]
    transcriptome_keys = [
        "expression_tokens",
        "expression_token_lengths",
        "expression_gene",
        "expression_expr",
        "expression_key_padding_mask",
    ]
    # Additional keys that should be preserved but not used for modality detection
    weight_keys = ["transcriptome_weights", "annotation_weights"]
    other_keys = ["orig_ids", "labels"]

    # Initialize modality masks
    text_mask = torch.zeros(batch_size, dtype=torch.bool)
    image_mask = torch.zeros(batch_size, dtype=torch.bool)
    transcriptome_mask = torch.zeros(batch_size, dtype=torch.bool)

    # Collect all keys present in the updated batch
    all_keys = set()
    for sample in batch:
        all_keys.update(sample.keys())

    # For each key, collate the values
    for key in all_keys:

        values = []
        key_present = []

        # Determine which modality this key belongs to
        is_text_key = key in text_keys
        is_image_key = key in image_keys
        is_transcriptome_key = key in transcriptome_keys
        is_weight_key = key in weight_keys
        is_other_key = key in other_keys

        for i, sample in enumerate(batch):
            if key in sample and sample[key] is not None:
                values.append(sample[key])
                key_present.append(True)

                # Update modality masks
                if is_text_key:
                    text_mask[i] = True
                elif is_image_key:
                    image_mask[i] = True
                elif is_transcriptome_key:
                    transcriptome_mask[i] = True
            else:
                key_present.append(False)

        # Handle weight keys and other special keys differently
        if is_weight_key or is_other_key:
            # For weight/other keys, just collect all values (can be different types)
            collated_values = []
            for sample in batch:
                if key in sample and sample[key] is not None:
                    collated_values.append(sample[key])
                else:
                    # For weights, default to 1.0 tensor, for others use None
                    if is_weight_key:
                        collated_values.append(torch.ones(1))  # Default weight
                    else:
                        collated_values.append(None)

            if is_weight_key and all(
                torch.is_tensor(v) for v in collated_values if v is not None
            ):
                try:
                    collated_batch[key] = torch.stack(
                        [v if v is not None else torch.ones(1) for v in collated_values]
                    )
                except RuntimeError:
                    collated_batch[key] = collated_values
            else:
                collated_batch[key] = collated_values
            continue

        if not values:
            # No samples have this key, skip it
            continue

        # Get representative shape/dtype from first valid sample
        if torch.is_tensor(values[0]):
            representative_tensor = values[0]
            dtype = representative_tensor.dtype
            shape = representative_tensor.shape
        else:
            # Handle non-tensor values (shouldn't happen with our data but for safety)
            collated_batch[key] = [sample.get(key) for sample in batch]
            continue

        # Create zero tensors for missing values and collate
        collated_values = []
        value_idx = 0

        for i, has_key in enumerate(key_present):
            if has_key:
                collated_values.append(values[value_idx])
                value_idx += 1
            else:
                zero_tensor = torch.zeros(shape, dtype=dtype)
                collated_values.append(zero_tensor)

        # Stack the tensors
        try:
            collated_batch[key] = torch.stack(collated_values)
        except RuntimeError as e:
            # If tensors have different shapes, we need padding
            if key == "expression_tokens":
                # Handle variable-length expression tokens by padding to max length in batch
                max_length = max(
                    tensor.shape[0] for tensor in collated_values if tensor.numel() > 0
                )
                padded_values = []
                for tensor in collated_values:
                    if tensor.numel() == 0:
                        # Handle empty tensors
                        padded_tensor = torch.zeros(max_length, dtype=tensor.dtype)
                    else:
                        current_length = tensor.shape[0]
                        if current_length < max_length:
                            padded_tensor = torch.nn.functional.pad(
                                tensor, (0, max_length - current_length), value=0
                            )
                        else:
                            padded_tensor = tensor
                    padded_values.append(padded_tensor)
                collated_batch[key] = torch.stack(padded_values)
            else:
                logger.error(f"Shape mismatch for key {key}: {e}")
                raise ValueError(f"Cannot handle shape mismatch for key {key}: {e}")

    # Add modality masks to the batch
    collated_batch["text_batch_mask"] = text_mask
    collated_batch["image_batch_mask"] = image_mask
    collated_batch["transcriptome_batch_mask"] = transcriptome_mask

    return collated_batch


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

        # Legacy support: rename "patches" key to "patches_ctx" if present
        if "patches" in sample:
            sample["patches_ctx"] = sample.pop("patches").squeeze(0)
            sample["patches_cell"] = torch.zeros((3, 56, 56), dtype=torch.float32)

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


class JointEmbedDatasetDisk(Dataset):
    """
    Dataset that loads samples from individual .pt files on disk.
    This is designed to handle large datasets that cannot fit in RAM.
    """

    def __init__(self, dataset_name: str, orig_ids, hash: str, i: Optional[str] = None):
        self.dataset_name = dataset_name
        self.orig_ids = orig_ids
        self.i = i if i is not None else ""
        self.hash = hash
        self.base_path = get_path(
            ["paths", "data_loading_individual_samples"],
            dataset_name=dataset_name,
            hash=self.hash,
            i=self.i,
        )

    def __len__(self):
        return len(self.orig_ids)

    def __getitem__(self, idx):
        orig_id = self.orig_ids[idx]
        # Construct the path to the individual .pt file
        sample_path = self.base_path / f"{orig_id}.pt"

        if not sample_path.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_path}")

        # Load the sample from disk
        sample = torch.load(sample_path, map_location="cpu")

        # Legacy support: rename "patches" key to "patches_ctx" if present
        if "patches" in sample:
            sample["patches_ctx"] = sample.pop("patches").squeeze(0)
            sample["patches_cell"] = torch.zeros((3, 56, 56), dtype=torch.float32)

        return sample


class JointEmbedDataModule(pl.LightningDataModule):
    """
    Generates training/validation datasets containing matching transcriptome-annotation pairs.

    Data is loaded from AnnData objects (.h5ad). For processing details refer to TranscriptomeTextDualEncoderProcessor
    """

    def __init__(
        self,
        tokenizer="bert",
        transcriptome_processor="geneformer",
        image_processor="uni2",
        dataset_names="human_disease",
        batch_size=32,
        nproc=8,
        transcriptome_processor_kwargs={},
        tokenizer_kwargs={
            "model_max_length": 128  # 128 seems to be a decent fit (previously 100)
        },  # see https://github.com/epigen/cellwhisperer/issues/193
        min_genes=100,
        train_fraction: Union[str, float] = 0.95,
        use_replicates: bool = False,
        include_labels: Optional[str] = None,
        use_disk_loading: bool = False,
        cosmx6k_filter: bool = False,
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
            use_disk_loading: whether to use disk-based loading (JointEmbedDatasetDisk) instead of loading all data into RAM.
                             Recommended for large datasets that cannot fit in memory.
            cosmx6k_filter: whether to filter genes to the CosMx 6K gene list before processing.
        """
        super().__init__()
        self.batch_size = batch_size
        self.dataset_names = dataset_names.split(",")
        self.tokenizer = model_path_from_name(tokenizer)
        self.transcriptome_processor = transcriptome_processor
        self.image_processor = image_processor
        self.nproc = int(nproc)
        self.min_genes = min_genes
        self.transcriptome_processor_kwargs = transcriptome_processor_kwargs.copy()
        self.tokenizer_kwargs = tokenizer_kwargs.copy()
        self.use_replicates = use_replicates
        self.include_labels = include_labels
        self.use_disk_loading = use_disk_loading
        self.cosmx6k_filter = cosmx6k_filter

        if self.cosmx6k_filter:
            gene_list_path = str(get_path(["paths", "cosmx6k_genes"]))
            self._cosmx6k_genes = set(pd.read_csv(gene_list_path)["gene_name"].tolist())

        self.train_fraction = train_fraction

        if isinstance(self.train_fraction, str):  # parse option for int/float
            try:
                self.train_fraction = int(train_fraction)
            except ValueError:
                try:
                    self.train_fraction = float(train_fraction)
                except ValueError:
                    pass

        # Initialize processor
        # Use CONCH tokenizer when requested, otherwise fall back to HF AutoTokenizer

        if self.tokenizer == "conch_text":
            tokenizer_obj = ConchTextProcessor()
        else:
            tokenizer_obj = (
                AutoTokenizer.from_pretrained(self.tokenizer, **self.tokenizer_kwargs)
                if self.tokenizer
                else None
            )

        self.processor = TranscriptomeTextDualEncoderProcessor(
            self.transcriptome_processor,
            tokenizer_obj,
            self.image_processor,
            nproc=self.nproc,
        )

    def _compute_hash(self, i: Optional[str] = None):
        """Compute hash from processing parameters"""
        return "_".join(
            [
                self.transcriptome_processor,
                "" if not self.tokenizer else self.tokenizer.replace("/", "__"),
                self.image_processor,
                str(self.min_genes),
                str(self.use_replicates),
                self.include_labels or "",
                "cosmx6k" if self.cosmx6k_filter else "",
                i or "",
            ]
        )

    def _processed_path(self, dataset_name, i: Optional[str] = None):
        return get_path(
            ["paths", "datamodule_prepared_path"],
            dataset=dataset_name,
            hash=self._compute_hash(i),
        )

    def get_sample_ids(self, dataset_name):
        """
        Return sample IDs for a given dataset, generate paths on-demand

        TODO: would be much cleaner to have a csv file (as already done for quilt1m) and take files from in there! (would need to implement for hest1k still, and provide coherent naming for the csv)
        """
        adata_path = get_path(["paths", "full_dataset"], dataset=dataset_name)

        if adata_path.exists():
            adata = anndata.read_h5ad(adata_path, backed="r")

            if "multi_sample_fns" in adata.uns:
                return adata.uns["multi_sample_ids"]
            else:
                return [""]  # Default empty sample_id for single file
        else:
            adata_paths = [
                Path(v)
                for v in glob.glob(
                    get_path(
                        ["paths", "full_dataset_multi"], dataset=dataset_name, i="*"
                    ).as_posix()
                )
            ]

            # if we are in test mode, then only use the first 2 datasets
            if self.trainer and self.trainer.fast_dev_run:
                adata_paths = adata_paths[:3]

            if not adata_paths:
                raise FileNotFoundError(
                    f"Neither full_data.h5ad, nor full_data_{{i}}.h5ad were found for {dataset_name}"
                )
            else:
                # Extract sample_ids from the 'full_data_{i}.h5ad' pattern
                return [
                    adata_path.stem.split("full_data_")[-1]
                    for adata_path in adata_paths
                ]

    def get_adata_path(self, dataset_name, sample_id):
        """Generate adata path from dataset name and sample ID"""
        if sample_id == "":
            return get_path(["paths", "full_dataset"], dataset=dataset_name)
        else:
            return get_path(
                ["paths", "full_dataset_multi"], dataset=dataset_name, i=sample_id
            )

    def get_processed_path(self, dataset_name, sample_id):
        """Generate processed path from dataset name and sample ID"""
        return self._processed_path(dataset_name, sample_id)

    def prepare_data(self, force_prepare=False):
        # Generate all datasets
        for dataset_name in self.dataset_names:
            self.prepare_dataset(dataset_name, force_prepare)

    def prepare_dataset(self, dataset_name, force_prepare):
        sample_ids = self.get_sample_ids(dataset_name)

        if not force_prepare:
            # check whether data has already been prepared
            filtered_sample_ids = []
            for sample_id in sample_ids:
                adata_path = self.get_adata_path(dataset_name, sample_id)
                processed_path = self.get_processed_path(dataset_name, sample_id)

                needs_preparation = False

                # Check if main processed file exists and is up to date
                if (
                    not processed_path.exists()
                    or processed_path.stat().st_mtime < adata_path.stat().st_mtime
                ):
                    needs_preparation = True
                else:
                    # checking for the individual files takes a couple of seconds, which is an overhead we are not willing to take (rather just crash the whole thing :))
                    if False:  # TODO disable for time reasons
                        # Check if individual sample files exist and are complete
                        individual_samples_dir = get_path(
                            ["paths", "data_loading_individual_samples"],
                            dataset_name=dataset_name,
                            hash=self._compute_hash(sample_id),
                            i=sample_id,
                        )

                        if not individual_samples_dir.exists():
                            needs_preparation = True
                        else:
                            # Load the processed file to get orig_ids and check if all individual files exist
                            try:
                                result = torch.load(str(processed_path), mmap=True)
                                orig_ids = result[0][
                                    "orig_ids"
                                ]  # result[0] is inputs dict, result[1] is replicate_inputs dict

                                # Check if all individual sample files exist
                                missing_files = [
                                    orig_id
                                    for orig_id in orig_ids
                                    if not (
                                        individual_samples_dir / f"{orig_id}.pt"
                                    ).exists()
                                ]

                                if missing_files:
                                    logger.info(
                                        f"Missing {len(missing_files)} individual sample files for {dataset_name} - {sample_id}"
                                    )
                                    needs_preparation = True
                            except Exception as e:
                                logger.warning(
                                    f"Error checking individual files for {dataset_name} - {sample_id}: {e}"
                                )
                                needs_preparation = True

                if needs_preparation:
                    filtered_sample_ids.append(sample_id)

            if not filtered_sample_ids:
                # If no files to process, return early
                logger.info(
                    f"No new data to prepare for {dataset_name}. Skipping preparation."
                )
                return

            sample_ids = filtered_sample_ids

        logger.info(f"Preparing {len(sample_ids)} data files for {dataset_name}...")
        for sample_id in sample_ids:
            adata_path = self.get_adata_path(dataset_name, sample_id)
            processed_path = self.get_processed_path(dataset_name, sample_id)
            self.prepare_dataset_file(
                adata_path, processed_path, dataset_name, sample_id
            )

    def prepare_dataset_file(self, adata_path, processed_path, dataset_name, sample_id):
        # Load data
        adata = anndata.read_h5ad(adata_path)

        # Filter genes to CosMx 6K gene list if requested
        if self.cosmx6k_filter:
            if "gene_name" in adata.var.columns:
                gene_names = adata.var["gene_name"].astype(str)
            else:
                gene_names = adata.var.index.astype(str)
            mask = gene_names.isin(self._cosmx6k_genes)
            adata = adata[:, mask].copy()
            logger.info(f"CosMx 6K filter: kept {mask.sum()}/{len(mask)} genes")

        # Fixed size embedding (https://huggingface.co/docs/transformers/en/pad_truncation), as we combine multiple datasets
        inputs = self.processor(
            text=(
                list(adata.obs["natural_language_annotation"])
                if self.tokenizer and "natural_language_annotation" in adata.obs
                else None
            ),
            transcriptomes=adata if adata.shape[1] > 0 else None,
            image=(
                adata
                if (
                    "he_slide" in adata.uns
                    or "image_path" in adata.uns
                    or "20x_slide" in adata.uns
                )  # 20x_slide is legacy for HEST
                and self.image_processor
                else None
            ),  # NOTE Could refactor API to only provide an adata, but not sure if the repository depends on this splitting..
            return_tensors="pt",
            padding="max_length",  # TODO maybe better to in collator? (but we need fixed size here to generate tensors and individual sample files)
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

        if self.use_replicates:
            logging.info("Preparing replicates")
            replicate_inputs = self._prepare_replicate_inputs(inputs, adata)
        else:
            replicate_inputs = {}

        if self.include_labels is not None:
            if adata.obs[self.include_labels].dtype.name != "category":
                adata.obs[self.include_labels] = pd.Categorical(
                    adata.obs[self.include_labels]
                )

            # Add labels to the inputs.
            inputs["labels"] = torch.tensor(
                adata.obs[self.include_labels].astype("category").cat.codes.values,
                dtype=torch.long,
            )

        # Filter for empty inputs (NOTE: empty inputs should be avoided)
        if (
            self.transcriptome_processor == "geneformer"
            and "expression_token_lengths" in inputs
        ):
            n_genes_filter = inputs["expression_token_lengths"] >= self.min_genes
        elif (
            self.transcriptome_processor == "scgpt"
            and "expression_key_padding_mask" in inputs
        ):
            n_genes_filter = (inputs["expression_key_padding_mask"] == False).sum(
                dim=1
            ) >= self.min_genes
        elif (
            self.transcriptome_processor.startswith("uce")
            and "expression_key_padding_mask" in inputs
        ):
            n_genes_filter = (inputs["expression_key_padding_mask"] == False).sum(
                dim=1
            ) >= 1  # self.min_genes  # NOTE the mask cannot be used unfortunately for filtering
        else:
            logging.debug("No gene values to filter on, using all samples")
            n_genes_filter = torch.ones(len(adata.obs), dtype=torch.bool)

        valid_datapoints_filter = n_genes_filter

        if self.image_processor.startswith("uni") and "patches_ctx" in inputs:
            # Check the context view for actual white/empty patches (not just negative sums)
            valid_patches = self._filter_invalid_patches(inputs["patches_ctx"])
            valid_datapoints_filter = valid_datapoints_filter & valid_patches

        # Filter cells that are too bright (only for datasets with brightness information)
        if "is_too_bright" in adata.obs.columns:
            brightness_filter = ~adata.obs["is_too_bright"].values
            n_too_bright = (~brightness_filter).sum()
            logger.info(
                f"Found 'is_too_bright' column: {n_too_bright}/{len(brightness_filter)} cells marked as too bright ({n_too_bright/len(brightness_filter)*100:.1f}%)"
            )
            valid_datapoints_filter = valid_datapoints_filter & torch.from_numpy(
                brightness_filter
            )

        if sum(valid_datapoints_filter) == len(valid_datapoints_filter):
            logger.info(
                f"No samples were filtered out (All cells had >= {self.min_genes} genes and/or non-white content)"
            )
            inputs["orig_ids"] = adata.obs.index[valid_datapoints_filter]
        else:
            logger.warning(
                f"Filtering for {sum(valid_datapoints_filter)} of {len(valid_datapoints_filter)} samples with >={self.min_genes} genes and valid image content"
            )
            # Apply filtering to inputs for all tensor keys
            filtered_inputs = {}
            for key, val in inputs.items():
                if key == "orig_ids":
                    continue
                filtered_inputs[key] = val[valid_datapoints_filter]
            inputs = filtered_inputs
            inputs["orig_ids"] = adata.obs.index[valid_datapoints_filter]
            if len(replicate_inputs) > 0:
                for key, rep_value in replicate_inputs.items():
                    replicate_inputs[key] = [
                        value[valid_datapoints_filter] for value in rep_value
                    ]
                    assert len(rep_value) > 0

        # process/generate the individual files
        self._generate_individual_sample_files(inputs, dataset_name, sample_id)

        # save the inputs dict to a file using torch (NOTE: In case of loading with JointEmbedDatasetDisk, this is largely redundant as we'd only need the orig_ids. But JointEmbedDataset still uses all from this file actually
        processed_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            (inputs, replicate_inputs),
            processed_path,
        )

    def _prepare_replicate_inputs(self, inputs, adata):
        """
        Take the length from the first one and apply it to all of them
        This is not required per se (the dimensionalities could also vary from epoch to epoch), however, it feels cleaner to remain dimensionalities across epochs.
        We could also stack them into a single tensor, but I see little benefit at the moment
        Note that the first one is computed twice (redundantly)
        """
        raise NotImplementedError(
            "Not impelemented for images. not sure how much there is to be done."
        )

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

    def _filter_invalid_patches(self, patches):
        """
        Filter out invalid patches based on standard deviation.

        Low standard deviation indicates uniform patches (empty, white background, etc.)
        This replaces the flawed sum-based filtering which incorrectly removed patches
        with color bias (e.g., reddish tissue with negative total sums).

        Args:
            patches (torch.Tensor): Context patches tensor of shape (N, 3, H, W)

        Returns:
            torch.Tensor: Boolean mask of shape (N,) indicating valid patches
        """
        # Check for patches with sufficient variation across pixels
        # Empty, white, or uniform patches will have very low standard deviation
        patch_std = patches.std(dim=(1, 2, 3))  # Standard deviation per patch
        std_threshold = 0.1  # Patches with std < 0.1 are likely uniform/invalid

        valid_patches = patch_std > std_threshold

        # Log filtering details
        total_patches = len(patches)
        filtered_count = (~valid_patches).sum().item()

        # TODO the threshold is way too low. Also, I redundantly implemented the all-white thresholding
        logger.info(
            f"Image filtering: {filtered_count}/{total_patches} patches filtered "
            f"(std < {std_threshold})"
        )

        return valid_patches

    def _generate_individual_sample_files(self, inputs, dataset_name, sample_id):
        logging.info(
            f"Generating individual sample files for {dataset_name} - {sample_id}"
        )

        # Create directory for individual sample files
        individual_samples_dir = get_path(
            ["paths", "data_loading_individual_samples"],
            dataset_name=dataset_name,
            hash=self._compute_hash(sample_id),
            i=sample_id,
        )
        individual_samples_dir.mkdir(parents=True, exist_ok=True)

        # Extract orig_ids for this batch
        orig_ids = inputs["orig_ids"]

        # Create individual .pt files for each sample (without global padding)
        for j, orig_id in enumerate(tqdm(orig_ids)):
            sample_file = individual_samples_dir / f"{orig_id}.pt"

            # Create individual sample dict
            sample = {}
            for key, tensor in inputs.items():
                if key == "orig_ids":
                    continue  # Skip orig_ids as it's metadata

                # Get the data for this specific sample (no padding applied here). Need to clone to prevent storing full tensors in case of views
                # Write tensors directly per key
                sample[key] = tensor[j].clone()

            # Save individual sample
            torch.save(sample, sample_file)

    def _load_disk_datasets(self, dataset_name):
        """Generate individual .pt files for each sample and return disk datasets"""
        sample_ids = self.get_sample_ids(dataset_name)
        disk_datasets = []

        # Process each file individually to avoid loading all into memory
        for sample_id in sample_ids:
            processed_path = self.get_processed_path(dataset_name, sample_id)

            # Load only this specific processed file
            result = torch.load(str(processed_path), mmap=True)

            # Create JointEmbedDatasetDisk for this batch
            disk_dataset = JointEmbedDatasetDisk(
                dataset_name=dataset_name,
                orig_ids=result[0]["orig_ids"],
                i=sample_id,
                hash=self._compute_hash(sample_id),
            )
            disk_datasets.append(disk_dataset)

        return disk_datasets

    def _load_processed_dataset_memory(self, dataset_name):
        """Original memory-based loading - aggregate and return"""
        processed_paths = [
            self.get_processed_path(dataset_name, sample_id)
            for sample_id in self.get_sample_ids(dataset_name)
        ]
        results = [torch.load(p, weights_only=False) for p in processed_paths]

        if len(results[0][1]) > 0:
            raise NotImplementedError("Currently, no 'replicates' are supported")

        # TODO this is not reflected in _load_disk_datasets! Perhaps better to do this in the collator?
        if "expression_token_lengths" in results[0][0]:
            # pad the expression tokens to the maximum length
            max_expression_length = (
                torch.cat([result[0]["expression_token_lengths"] for result in results])
                .max()
                .item()
            )
            for result in results:
                # pad the expression token lengths to the maximum length
                result[0]["expression_tokens"] = torch.nn.functional.pad(
                    result[0]["expression_tokens"],
                    (
                        0,
                        max_expression_length - result[0]["expression_tokens"].shape[1],
                    ),
                    value=0,
                )

        # Aggregate results
        aggregated_inputs = {}
        for key in results[0][0]:
            if key == "orig_ids":
                aggregated_inputs[key] = pd.Index(
                    [v for result in results for v in result[0][key]]
                )
            else:
                aggregated_inputs[key] = torch.cat(
                    [result[0][key] for result in results], dim=0
                )

        return (
            aggregated_inputs,
            {
                # replicates not supported
            },
        )

    def setup(self, stage=None):
        self.train_datasets = []
        self.val_datasets = []

        for dataset_name in self.dataset_names:
            # TODO: Refactor memory loading to follow the same logic as disk loading (keeping datasets separate)
            # instead of concatenating everything into single tensors. This would make both paths more consistent.
            if self.use_disk_loading:
                # Generate individual sample files and get disk datasets
                disk_datasets = self._load_disk_datasets(dataset_name)

                # For each disk dataset, split into train/val based on train_fraction
                for disk_dataset in disk_datasets:

                    dataset_len = len(disk_dataset.orig_ids)

                    if isinstance(self.train_fraction, (int, float)):
                        # Assuming you want to split the data into train and val for simplicity
                        train_size = int(self.train_fraction * dataset_len)
                        # randomly sample train_size indices for train and use the rest for val
                        # fix the seed
                        random.seed(42)
                        total_ids = list(range(dataset_len))
                        train_ids = random.sample(total_ids, train_size)
                        val_ids = sorted(list(set(total_ids) - set(train_ids)))
                    elif isinstance(self.train_fraction, str):
                        total_ids = list(range(dataset_len))
                        if dataset_name in self.train_fraction.split(","):
                            train_ids = total_ids
                            val_ids = []
                        else:
                            train_ids = []
                            val_ids = total_ids
                    else:
                        raise ValueError(
                            "train_fraction must be either a float or string"
                        )

                    # Create train dataset if we have train samples
                    if train_ids:
                        train_disk_dataset = JointEmbedDatasetDisk(
                            dataset_name=disk_dataset.dataset_name,
                            orig_ids=disk_dataset.orig_ids[train_ids],
                            i=disk_dataset.i,
                            hash=disk_dataset.hash,
                        )
                        self.train_datasets.append(train_disk_dataset)

                    # Create val dataset if we have val samples
                    if val_ids:
                        val_disk_dataset = JointEmbedDatasetDisk(
                            dataset_name=disk_dataset.dataset_name,
                            orig_ids=disk_dataset.orig_ids[val_ids],
                            i=disk_dataset.i,
                            hash=disk_dataset.hash,
                        )
                        self.val_datasets.append(val_disk_dataset)
            else:
                # Use original memory-based loading
                (inputs, replicate_inputs) = self._load_processed_dataset_memory(
                    dataset_name
                )

                # Use orig_ids length to determine dataset size
                dataset_len = len(inputs["orig_ids"])
                if isinstance(self.train_fraction, (int, float)):
                    # Assuming you want to split the data into train and val for simplicity
                    train_size = int(self.train_fraction * dataset_len)
                    # randomly sample train_size indices for train and use the rest for val
                    # fix the seed
                    random.seed(42)
                    total_ids = list(range(dataset_len))
                    train_ids = random.sample(total_ids, train_size)
                    val_ids = sorted(list(set(total_ids) - set(train_ids)))
                elif isinstance(self.train_fraction, str):
                    total_ids = list(range(dataset_len))
                    if dataset_name in self.train_fraction.split(","):
                        train_ids = total_ids
                        val_ids = []
                    else:
                        train_ids = []
                        val_ids = total_ids
                else:
                    raise ValueError("train_fraction must be either a float or string")

                # Build train inputs with support for nested image patches dict
                train_inputs = {}
                for key, value in inputs.items():
                    if key == "orig_ids":
                        continue
                    train_inputs[key] = value[train_ids]

                self.train_datasets.append(
                    JointEmbedDataset(
                        train_inputs,
                        orig_ids=inputs["orig_ids"][train_ids],
                        replicate_inputs={
                            key: [value[i][train_ids] for i in range(len(value))]
                            for key, value in replicate_inputs.items()
                            if len(value) > 0
                        },
                    )
                )

                # Build val inputs with support for nested image patches dict
                val_inputs = {}
                for key, value in inputs.items():
                    if key == "orig_ids":
                        continue
                    val_inputs[key] = value[val_ids]

                self.val_datasets.append(
                    JointEmbedDataset(
                        val_inputs,
                        orig_ids=inputs["orig_ids"][val_ids],
                    )
                )

    def train_dataloader(self):
        # Update the current epoch to sample the replicates (only for memory-based loading)
        if self.use_replicates and not self.use_disk_loading:
            for dataset in self.train_datasets:
                if hasattr(dataset, "set_epoch"):  # JointEmbedDataset has this method
                    dataset.set_epoch(self.trainer.current_epoch)

        return DataLoader(
            ConcatDataset(self.train_datasets),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.nproc,
            persistent_workers=self.nproc > 0,  # Keep workers alive to avoid hanging
            drop_last=True,  # drop last batch to avoid batch_size of 1, which fails due to batch-norm
            collate_fn=multi_modal_collate_fn,  # Use custom collator for multi-modal data
        )

    def val_dataloader(self, shuffle=False):
        """
        In principle, we could also return a list of DataLoaders, but it is currently incompatible with RetrievalScoreCalculator
        """
        return DataLoader(
            ConcatDataset(self.val_datasets),
            batch_size=self.batch_size,
            num_workers=self.nproc,
            persistent_workers=self.nproc > 0,  # Keep workers alive to avoid hanging
            drop_last=False,  # more accurate if we don't drop the last
            shuffle=shuffle,
            collate_fn=multi_modal_collate_fn,  # Use custom collator for multi-modal data
        )

    def test_dataloader(self):
        # Return the validation dataloader for testing
        return self.val_dataloader(shuffle=False)

    def predict_dataloader(self):
        """
        Return dataloader for prediction.
        Uses val_datasets if available, otherwise uses train_datasets (for train_fraction=1.0).
        """
        # Use validation data if available, otherwise use training data

        return DataLoader(
            ConcatDataset(self.train_datasets),
            batch_size=self.batch_size,
            num_workers=self.nproc,
            persistent_workers=self.nproc > 0,
            drop_last=False,
            shuffle=False,  # Never shuffle for prediction
            collate_fn=multi_modal_collate_fn,
        )
