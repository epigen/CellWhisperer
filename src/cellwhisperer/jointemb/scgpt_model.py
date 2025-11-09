from cellwhisperer.config import get_path
import torch
import scipy.sparse as sp
import numpy as np
import dataclasses
import pandas as pd
import logging
import torch.nn.functional as F

from transformers import BertForMaskedLM, BertConfig
from typing import Optional, Union, Tuple, Any
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.configuration_utils import PretrainedConfig
from transformers.processing_utils import ProcessorMixin
from pathlib import Path

try:
    from scgpt.tokenizer import GeneVocab
    from scgpt.data_collator import DataCollator
    from scgpt.model import TransformerModel
    from scgpt import logger
    from scgpt.utils import load_pretrained
except ModuleNotFoundError:
    logging.warning("scGPT not installed (only required if you want to use it).")

from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import os
from pathlib import Path
import json
import numpy as np
from anndata import AnnData
import scanpy as sc
import inspect

import anndata
import scanpy as sc


class ScGPTDataset(torch.utils.data.Dataset):
    """
    Adapted from: https://github.com/bowang-lab/scGPT/blob/418b0f623fb1f17641a12c9e50f72f4419311745/scgpt/tasks/cell_emb.py#L22
    """

    def __init__(self, count_matrix, gene_ids, vocab, pad_value, batch_ids=None):
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.batch_ids = batch_ids
        self.vocab = vocab
        self.pad_value = pad_value

    def __len__(self):
        return len(self.count_matrix)

    def __getitem__(self, idx):
        row = self.count_matrix[idx]
        nonzero_idx = np.nonzero(row)[
            0
        ]  # this is fishy somehow. <row> can be a float with values such as 0.3. they are nonzero, but become 0 when casted to int
        values = row[nonzero_idx].astype(
            np.int64
        )  # uint row matrixes cannot represent pad_token (-2)
        if values.max() == 0:
            values = np.ceil(row[nonzero_idx]).astype(np.int64)

        genes = self.gene_ids[nonzero_idx]
        # append <cls> token at the beginning
        genes = np.insert(genes, 0, self.vocab["<cls>"])
        values = np.insert(values, 0, self.pad_value)
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values)
        output = {
            "id": idx,
            "genes": genes,
            "expressions": values,
        }
        if self.batch_ids is not None:
            output["batch_labels"] = self.batch_ids[idx]
        return output


class ScGPTTranscriptomeProcessor(ProcessorMixin):
    attributes = []

    def __init__(
        self,
        *args,
        adata_filter_genes_by_count_min_count=None,
        adata_filter_cells_by_count_min_count=None,
        adata_do_normalize_total=True,
        adata_do_log1p=True,
        adata_n_hvgs=None,
        adata_hvg_flavor="cell_ranger",
        gene_col="gene_name",
        vocab_path=str(get_path(["model_name_path_map", "scgpt"]) / "vocab.json"),
        pad_token="<pad>",
        nproc=10,
        **kwargs,
    ):
        """
        Tokenizer for scGPT model.
        Note that the tokenizer differentiates between genes with zero- and nonzero counts, and this will only work if the input data is raw counts, \
        and adata_do_normalize_total and adata_do_log1p are False. Alternatively, hvg selection can be used.

        Args:
            adata_filter_genes_by_count_min_count (int): The minimum number of counts for a gene to be included. Defaults to None (no filtering).
            adata_filter_cells_by_count_min_count (int): The minimum number of counts for a cell to be included. Defaults to None (no filtering).
            adata_do_normalize_total (bool): Whether to normalize the total counts (to 1e4). Defaults to False.
            adata_do_log1p (bool): Whether to log1p transform the counts. Defaults to False.
            adata_n_hvgs (int): The number of highly variable genes to select. Defaults to None (no selection).
            adata_hvg_flavor (str): The flavor of highly variable gene selection. Defaults to "seurat_v3". Expects logarithmized data, except when flavor='seurat_v3', in which count data is expected. Ignored if adata_n_hvgs is None.
            gene_col (str): The column name of the gene names in the anndata object. Defaults to "gene_name".
            vocab_path (str): The path to the vocabulary file. Default: Use get_path(["model_name_path_map", "scgpt"]) / "vocab.json". Unused/deprecated
            pad_token (str): The padding token. Defaults to "<pad>".
            nproc (int): The number of processes to use for tokenization. Defaults to 10.
        """

        self.adata_filter_genes_by_count_min_count = (
            adata_filter_genes_by_count_min_count
        )
        self.adata_filter_cells_by_count_min_count = (
            adata_filter_cells_by_count_min_count
        )
        self.adata_do_normalize_total = adata_do_normalize_total
        self.adata_do_log1p = adata_do_log1p
        self.adata_n_hvgs = adata_n_hvgs
        self.adata_hvg_flavor = adata_hvg_flavor

        self.vocab_path = str(
            get_path(["model_name_path_map", "scgpt"]) / "vocab.json"
        )  # unused/deprecated
        self.pad_token = pad_token
        self.gene_col = gene_col
        self.nproc = 0

        self.vocab = load_vocab(self.pad_token)

        super().__init__(*args, **kwargs)

    def _tokenize(
        self,
        adata,
        max_length=1200,
        batch_size=64,
        pad_token="<pad>",
        pad_value=-2,
        gene_ids=None,
        use_batch_labels=False,
    ) -> torch.Tensor:
        """
        Tokenize the input data.
        Adapted from: https://github.com/bowang-lab/scGPT/blob/418b0f623fb1f17641a12c9e50f72f4419311745/scgpt/tasks/cell_emb.py#L22

        Args:
            adata (AnnData): The AnnData object.
            max_length (int): The maximum length of the input sequence. Defaults to 1200.
            batch_size (int): The batch size for inference. Defaults to 64.
            pad_token (str): The padding token. Defaults to "<pad>".
            pad_value (int): The padding value. Defaults to -2.
            gene_ids (np.ndarray, optional): The gene vocabulary ids. Defaults to None.
            use_batch_labels (bool): Whether to use batch labels. Defaults to False.

        Returns:
            torch.Tensor: The tokenized cells as tensor.
        """

        count_matrix = adata.X
        count_matrix = (
            count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
        )

        if gene_ids is None:
            gene_ids = np.array(adata.var["id_in_vocab"])
            assert np.all(gene_ids >= 0)

        if use_batch_labels:
            batch_ids = np.array(adata.obs["batch_id"].tolist())

        dataset = ScGPTDataset(
            count_matrix,
            gene_ids,
            self.vocab,
            batch_ids=batch_ids if use_batch_labels else None,
            pad_value=pad_value,
        )
        collator = DataCollator(
            do_padding=True,
            pad_token_id=self.vocab[pad_token],
            pad_value=pad_value,
            do_mlm=False,
            do_binning=True,
            max_length=max_length,
            sampling=True,
            keep_first_n_tokens=1,
        )
        # We need to process all at once. Otherwise the token lengths might vary across batches which leads to crashes (this is what padding is for...). If this leads to OOMs, we need to find another solution.
        data_loader = DataLoader(
            dataset,
            batch_size=len(dataset),
            sampler=SequentialSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            num_workers=min(self.nproc, batch_size),
            pin_memory=True,
        )

        output = {
            "expression_gene": [],
            "expression_expr": [],
            "expression_key_padding_mask": [],
        }
        if use_batch_labels:
            output["batch_labels"] = []

        assert len(data_loader) == 1, "We need to process all at once"
        data_dict = next(iter(data_loader))
        output["expression_gene"].append(data_dict["gene"])
        output["expression_expr"].append(data_dict["expr"])
        output["expression_key_padding_mask"].append(
            data_dict["gene"].eq(self.vocab[self.pad_token])
        )
        if use_batch_labels:
            output["batch_labels"].append(data_dict["batch_labels"])

        for key, value in output.items():
            output[key] = torch.cat(value, dim=0)

        if use_batch_labels:
            output["batch_labels"] = torch.cat(output["batch_labels"], dim=0)

        return output

    def _preprocess_adata(self, adata):
        """
        Preprocess the AnnData object. Adapted from: https://github.com/bowang-lab/scGPT/blob/418b0f623fb1f17641a12c9e50f72f4419311745/scgpt/preprocess.py#L13
        Args:
            adata (AnnData): The AnnData object.
        Returns:
            AnnData: The preprocessed AnnData object.
        """
        if any(
            [
                x is not None
                for x in [
                    self.adata_filter_genes_by_count_min_count,
                    self.adata_filter_cells_by_count_min_count,
                    self.adata_n_hvgs,
                ]
            ]
            + [self.adata_do_normalize_total, self.adata_do_log1p]
        ):
            adata = adata.copy()

        if self.adata_filter_genes_by_count_min_count is not None:
            logger.info(
                f"Filtering genes with less than {self.adata_filter_genes_by_count_min_count} counts"
            )
            sc.pp.filter_genes(
                adata, min_counts=self.adata_filter_genes_by_count_min_count
            )
        if self.adata_filter_cells_by_count_min_count is not None:
            logger.info(
                f"Filtering cells with less than {self.adata_filter_cells_by_count_min_count} counts"
            )
            sc.pp.filter_cells(
                adata, min_counts=self.adata_filter_cells_by_count_min_count
            )
        if self.adata_do_normalize_total:
            logger.info("Normalizing total counts to 1e4")
            sc.pp.normalize_total(adata, target_sum=1e4)
        if self.adata_do_log1p:
            logger.info("Log1p transforming counts")
            sc.pp.log1p(adata)
        if self.adata_n_hvgs is not None:
            logger.info(f"Selecting {self.adata_n_hvgs} highly variable genes")
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=self.adata_n_hvgs,
                subset=True,
                flavor=self.adata_hvg_flavor,
            )

        # make sure that there are no nans
        adata.var[self.gene_col] = np.where(
            adata.var[self.gene_col].isna(), adata.var.index, adata.var[self.gene_col]
        )
        return adata

    def __call__(self, adata, *args, **kwargs) -> dict:
        """
        Tokenize the input data.  \
         Also adapted from: https://github.com/bowang-lab/scGPT/blob/418b0f623fb1f17641a12c9e50f72f4419311745/scgpt/tasks/cell_emb.py#L22
        Args:
            adata (AnnData): The AnnData object to tokenize.
            *args: Additional arguments (ignored)
            **kwargs: Additional keyword arguments (ignored)
        Returns:
            dict: The tokenized data, as a dict of tensors.

        """
        adata = self._preprocess_adata(adata)

        adata.var["id_in_vocab"] = [
            self.vocab[gene] if gene in self.vocab else -1
            for gene in adata.var[self.gene_col]
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        logger.debug(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(self.vocab)}."
        )

        # Avoid to create views of views (as there was a bug in anndata)
        adata_view = adata[:, adata.var["id_in_vocab"] >= 0]

        # Identify rows with zero reads
        selector = adata_view.X.sum(axis=1) < 1
        if selector.sum() > 0:
            # Workaround for requirement in `Collator` to have at least one read-count gene per cell.
            # As we perform filtering for `min_genes`, this will be filtered anyways
            logging.info(
                f"There are {selector.sum()} rows with 0 read counts. Adding 1 read count to max expressed gene"
            )
            max_gene_idx = adata_view.X.sum(axis=0).argmax()
            adata_view.X[np.squeeze(np.asarray(selector)), max_gene_idx] = 1
        else:
            adata = adata_view

        genes = adata.var[self.gene_col].tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)

        result = self._tokenize(
            adata=adata,
            max_length=1200,
            batch_size=64,
            pad_token=self.pad_token,
            pad_value=-2,  # NOTE hardcoded here, but flexible elsewhere in this file
            gene_ids=gene_ids,
            use_batch_labels=False,
        )

        return result

    @property
    def model_input_names(self):
        return ["expression_gene", "expression_expr", "expression_key_padding_mask"]


class ScGPTConfig(PretrainedConfig):
    model_type = "scgpt"

    def __init__(
        self,
        pad_token="<pad>",
        input_emb_style="continuous",
        vocab_path=str(get_path(["model_name_path_map", "scgpt"]) / "vocab.json"),
        fast_transformer=False,
        nlayers=12,
        nheads=8,
        embsize=512,
        hidden_size=512,  # PP: renamed from d_hid
        dropout=0.2,
        n_layers_cls=3,
        pad_value=-2,
        gene_col="gene_name",
        device="cuda",
        n_cls=1,
        do_mvc=True,  # If yes, use an MVCDecoder (Decoder for masked value prediction for cell embeddings). True for cell embeddings, True for finetune integration, False for multiomic
        do_dab=False,  # If yes, use a AdversarialDiscriminator (grad_reverse_discriminator). False for cell embeddings, True for finetune integration, False for multiomic
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        fast_transformer_backend="flash",
        pre_norm=False,
        n_input_bins=None,
        cell_emb_style="cls",
        mvc_decoder_style="inner product",
        ecs_threshold=0.3,
        normalize_features=False,  # NOTE: In the cell embedding task, they do normalize the features, but it seems they didn't normalize during training.
        **kwargs,
    ):
        """
        Configuration for scGPT model.
        Adapted from the args for the scgpt.model.TransformerModel class, and set the defaults accoording to the scGPT embeddings example (https://github.com/bowang-lab/scGPT/blob/418b0f623fb1f17641a12c9e50f72f4419311745/scgpt/tasks/cell_emb.py#L237)
        """
        super().__init__(**kwargs)
        self.model_type = ScGPTConfig.model_type

        self.pad_token = pad_token
        self.input_emb_style = input_emb_style
        self.vocab_path = str(
            get_path(["model_name_path_map", "scgpt"]) / "vocab.json"
        )  # unused/workaround
        self.fast_transformer = fast_transformer
        self.nlayers = nlayers
        self.nheads = nheads
        self.embsize = embsize
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_layers_cls = n_layers_cls
        self.pad_value = pad_value
        self.gene_col = gene_col
        self.device = device
        self.n_cls = n_cls
        self.do_mvc = do_mvc
        self.do_dab = do_dab
        self.use_batch_labels = use_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.explicit_zero_prob = explicit_zero_prob
        self.fast_transformer_backend = fast_transformer_backend
        self.pre_norm = pre_norm
        self.n_input_bins = n_input_bins
        self.cell_emb_style = cell_emb_style
        self.mvc_decoder_style = mvc_decoder_style
        self.ecs_threshold = ecs_threshold
        self.normalize_features = normalize_features


def load_vocab(pad_token):
    """
    Load the vocabulary for scGPT. Adapted from https://github.com/bowang-lab/scGPT/blob/418b0f623fb1f17641a12c9e50f72f4419311745/scgpt/tasks/cell_emb.py#L148
    """
    # LOAD VOCAB
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    vocab = GeneVocab.from_file(
        str(get_path(["model_name_path_map", "scgpt"]) / "vocab.json")
    )
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab[pad_token])
    return vocab


class ScGPTModel(PreTrainedModel):
    def __init__(self, config):
        """
        The scGPT model. Adapted from https://github.com/bowang-lab/scGPT/blob/418b0f623fb1f17641a12c9e50f72f4419311745/scgpt/tasks/cell_emb.py#L22 \
        and https://github.com/bowang-lab/scGPT/blob/418b0f623fb1f17641a12c9e50f72f4419311745/scgpt/tasks/cell_emb.py#L148
        Args:
            config (ScGPTConfig): The configuration for the model.
        """
        #  model_dir: str, max_length: int = 1200,
        #  batch_size: int = 64, use_fast_transformer: bool = False,
        #  device: str = "cuda", gene_col: str = "gene_name"):

        super().__init__(config)
        self.config = config

        config_class = ScGPTConfig
        base_model_prefix = "scgpt_model"
        is_parallelizable = False  # not sure actually
        main_input_name = (
            "expression_gene"  # there are actually three main inputs, but good enough
        )

        # LOAD VOCAB
        self.vocab = load_vocab(self.config.pad_token)

        # CREATE MODEL
        allowed_args = list(inspect.signature(TransformerModel).parameters.keys())
        scgpt_model_kwargs = {
            k: v for k, v in config.to_dict().items() if k in allowed_args
        }

        # NOTE: These settings are adapted from the scGPT embeddings example.
        self.scgpt_model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.config.embsize,
            nhead=self.config.nheads,
            nlayers_cls=self.config.n_layers_cls,
            vocab=self.vocab,
            use_fast_transformer=self.config.fast_transformer,
            d_hid=self.config.hidden_size,
            **scgpt_model_kwargs,
        )
        if self.config.fast_transformer:
            # flash-attention requires 16bit apparently
            self.scgpt_model.to(config.device, dtype=torch.bfloat16)

    def forward(
        self,
        expression_gene,
        expression_expr,
        expression_key_padding_mask,
        expression_tokens: torch.Tensor = None,  # ignored, needed for compatibility
        expression_token_lengths: torch.Tensor = None,  # ignored, needed for compatibility
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the model. Obtain cell embeddings (i.e. features for the CLIP model) from tokenized input data.
        """
        # Convert to float16 for flash attention
        if self.config.fast_transformer:
            expression_expr = expression_expr.to(torch.bfloat16)
        else:
            expression_expr = expression_expr.to(torch.float)

        features = self.scgpt_model._encode(
            expression_gene,
            expression_expr,
            src_key_padding_mask=expression_key_padding_mask,
        )
        # get the <cls> position embedding and convert back to float32
        features = features[:, 0, :]

        # Convert back to float32
        if self.config.fast_transformer:
            features.to(torch.float)

        if self.config.normalize_features:
            features = F.normalize(features, p=2, dim=1)

        return (None, features)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path: str, *args, **kwargs
    ) -> PreTrainedModel:
        if "config" in kwargs:
            config = kwargs.pop("config")
            if isinstance(config, dict):
                config = ScGPTConfig(**config)
            elif not isinstance(config, ScGPTConfig):
                raise ValueError(
                    "Parameter `config` must be a dictionary or an instance of `ScGPTConfig`."
                )
        else:
            config = ScGPTConfig()
            logger.warning(
                "No configuration provided. Using default configuration from checkpoint."
            )

        outer_scgpt_model = cls(config, *args, **kwargs)

        # A scgpt function:
        outer_scgpt_model.scgpt_model = load_pretrained(
            outer_scgpt_model.scgpt_model,
            torch.load(pretrained_model_path),
            verbose=False,
        )

        return outer_scgpt_model
