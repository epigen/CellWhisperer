"""
UCE model wrapper using Kuan's data_collection_exp implementation.

Model checkpoint: KuanP/uce-cosmx-geneset (auto-downloaded from HuggingFace)
Gene data: resources/UCE/gene_names.txt and resources/UCE/all_species_gene_dict.json
  (originally from https://drive.google.com/drive/folders/1pWHRSqm3Njz1KPRfMP-ISNts785I2COG)
"""

from typing import Optional, List, Dict
import json
import logging

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader

from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.processing_utils import ProcessorMixin

from data_collection_suite.model.uce import (
    UCEConfig as KuanUCEConfig,
    UCEForExpressionPrediction,
)
from data_collection_suite.data.collator import UCEDataCollator
from data_collection_suite.data.cell_sentence_sampler import sample_cell_sentences_mapping_gene

from cellwhisperer.config import get_path

TARGET_SUM = 1e4


class _CellDataset(torch.utils.data.Dataset):
    """
    Lightweight Dataset that tokenizes cells from an AnnData for the new UCE model.
    Adapted from pert_lang_dataset/analysis/uce_embeddings/embed.py (NormalizedH5ADDataset).
    """

    def __init__(
        self,
        adata: sc.AnnData,
        gene_names: List[str],
        gene_mapping: Dict,
        pad_length: int = 2048,
        sample_size: int = 1024,
        species: str = "human",
    ):
        self.adata = adata
        self.gene_names = gene_names
        self.gene_mapping = gene_mapping
        self.pad_length = pad_length
        self.sample_size = sample_size

        # Tokenizer params (from KuanP/uce-cosmx-geneset config)
        self.cls_token_idx = 1
        self.chrom_token_offset = 1000
        self.chrom_token_right_idx = 2000
        self.pad_token_idx = 0
        self.positive_sample_num = 100
        self.negative_sample_num = 100
        self.mask_prop = 0.0  # no masking for embedding extraction

        # Build gene alignment
        self._align_genes(species)

    def _align_genes(self, species: str):
        """Align adata genes to the UCE vocabulary."""
        # Use gene_name column if available (for datasets like cellxgene_census with Ensembl IDs as index)
        if "gene_name" in self.adata.var.columns:
            h5ad_gene_names = list(self.adata.var["gene_name"].astype(str))
        else:
            h5ad_gene_names = list(self.adata.var_names)
        target_gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}
        mapping = self.gene_mapping

        valid_h5ad_indices = []
        gene_protein_ids = []
        gene_chroms = []
        gene_starts = []

        for h5ad_idx, gene in enumerate(h5ad_gene_names):
            if gene in target_gene_to_idx and gene in mapping:
                valid_h5ad_indices.append(h5ad_idx)
                m = mapping[gene]
                gene_protein_ids.append(m["protein_embedding_id"])
                gene_chroms.append(m["chromosome_id"])
                gene_starts.append(m["location"])

        self.valid_h5ad_indices = np.array(valid_h5ad_indices, dtype=np.int64)
        self.gene_protein_ids = np.array(gene_protein_ids, dtype=np.int64)
        self.gene_chroms = np.array(gene_chroms, dtype=np.int64)
        self.gene_starts = np.array(gene_starts, dtype=np.int64)

        logging.info(
            f"UCE gene alignment: {len(h5ad_gene_names)} h5ad genes, "
            f"{len(self.gene_names)} vocab genes, {len(self.valid_h5ad_indices)} in common"
        )

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        cell_expr = self.adata.X[idx]
        if hasattr(cell_expr, "toarray"):
            cell_expr = cell_expr.toarray().flatten()
        else:
            cell_expr = np.array(cell_expr).flatten()

        # Library-size normalize to TARGET_SUM
        total = cell_expr.sum()
        if total > 0:
            cell_expr = cell_expr / total * TARGET_SUM

        valid_expr = cell_expr[self.valid_h5ad_indices]
        counts_batch = torch.from_numpy(valid_expr.astype(np.float32)).unsqueeze(0)

        log_expr = torch.log1p(counts_batch)
        expr_sum = torch.clamp(log_expr.sum(dim=1, keepdim=True), min=1e-8)
        weights_batch = log_expr / expr_sum

        result = sample_cell_sentences_mapping_gene(
            counts=counts_batch,
            batch_weights=weights_batch,
            gene_protein_ids=self.gene_protein_ids,
            gene_chroms=self.gene_chroms,
            gene_starts=self.gene_starts,
            pad_length=self.pad_length,
            positive_sample_num=self.positive_sample_num,
            negative_sample_num=self.negative_sample_num,
            mask_prop=self.mask_prop,
            sample_size=self.sample_size,
            cls_token_idx=self.cls_token_idx,
            chrom_token_offset=self.chrom_token_offset,
            chrom_token_right_idx=self.chrom_token_right_idx,
            pad_token_idx=self.pad_token_idx,
            seed=idx,
        )

        result["idx"] = idx
        result["dataset_num"] = 0
        return result


def _load_gene_data(gene_names_path: str, gene_mapping_path: str, species: str = "human"):
    """Load gene names and gene mapping from files."""
    with open(gene_names_path, "r") as f:
        gene_names = [line.strip() for line in f.readlines()]
    with open(gene_mapping_path, "r") as f:
        gene_mapping = json.load(f)[species]
    return gene_names, gene_mapping


class UCETranscriptomeProcessor(ProcessorMixin):
    """
    Preprocessor that tokenizes AnnData into input_ids + attention_mask
    for the new Kuan UCE model.
    """

    attributes = []

    def __init__(self, nproc=8, **kwargs):
        self.nproc = nproc
        self.gene_names, self.gene_mapping = _load_gene_data(
            str(get_path(["uce_paths", "gene_names_path"])),
            str(get_path(["uce_paths", "gene_mapping_path"])),
        )
        super().__init__(**kwargs)

    def __call__(self, adata, *args, **kwargs) -> dict:
        """
        Tokenize an AnnData object into input_ids and attention_mask tensors.

        Returns:
            dict with keys "expression_expr" (input_ids) and
            "expression_key_padding_mask" (attention_mask).
        """
        dataset = _CellDataset(
            adata=adata,
            gene_names=self.gene_names,
            gene_mapping=self.gene_mapping,
        )
        collator = UCEDataCollator()
        dataloader = DataLoader(
            dataset,
            batch_size=1024,
            shuffle=False,
            collate_fn=collator,
            num_workers=min(self.nproc, len(dataset)),
        )
        all_input_ids = []
        all_attention_mask = []
        for batch in dataloader:
            all_input_ids.append(batch["input_ids"])
            all_attention_mask.append(batch["attention_mask"])

        return {
            "expression_expr": torch.cat(all_input_ids, dim=0),
            "expression_key_padding_mask": torch.cat(all_attention_mask, dim=0),
        }

    @property
    def model_input_names(self):
        return ["expression_expr", "expression_key_padding_mask"]


class UCEConfig(PretrainedConfig):
    """
    Cellwhisperer-compatible config wrapper for the new Kuan UCE model.
    Delegates to data_collection_suite.model.uce.UCEConfig for actual params.
    """

    model_type = "uce"

    def __init__(
        self,
        output_embedding_dim: int = 512,
        d_model: int = 512,
        nhead: int = 4,
        num_layers: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ):
        # output_dim is what cellwhisperer's discriminator reads for the projection size
        self.output_dim = output_embedding_dim
        self.output_embedding_dim = output_embedding_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        # Store all kwargs for passing through to Kuan's config
        self._kuan_kwargs = dict(
            output_embedding_dim=output_embedding_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            **{k: v for k, v in kwargs.items() if k not in ("model_type",)},
        )
        super().__init__(**kwargs)

    def to_kuan_config(self) -> KuanUCEConfig:
        """Convert to the underlying Kuan UCEConfig."""
        return KuanUCEConfig(**self._kuan_kwargs)


class UCEModel(PreTrainedModel):
    """
    Cellwhisperer-compatible UCE model wrapping Kuan's UCEForExpressionPrediction.

    The forward method accepts expression_expr/expression_key_padding_mask (cellwhisperer convention)
    and returns (gene_output, cell_embedding) tuple when return_dict=False, compatible with the
    existing discriminator pipeline which does `model(..., return_dict=False)[1]`.
    """

    config_class = UCEConfig
    base_model_prefix = "uce_model"

    def __init__(self, config: UCEConfig):
        super().__init__(config)
        self.config = config
        # The inner model will be loaded in from_pretrained; initialize with dummy
        self.inner_model: Optional[UCEForExpressionPrediction] = None

    def forward(
        self,
        expression_expr: torch.Tensor,
        expression_key_padding_mask: torch.Tensor,
        expression_tokens: Optional[torch.Tensor] = None,
        expression_token_lengths: Optional[torch.Tensor] = None,
        expression_gene: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """
        Forward pass. Adapts cellwhisperer's naming to Kuan's model API.

        expression_expr -> input_ids
        expression_key_padding_mask -> attention_mask

        Returns:
            If return_dict=False: (None, cell_embedding) — gene_output is None for compat
            If return_dict=True: UCEModelOutput
        """
        outputs = self.inner_model.uce.extract_cell_embeddings(
            input_ids=expression_expr,
            attention_mask=expression_key_padding_mask,
            return_dict=True,
        )

        if return_dict:
            return outputs

        # Return (gene_output, cell_embedding) for backward compat: callers do [1]
        return (outputs.gene_embeddings, outputs.cell_embedding)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs) -> "UCEModel":
        """
        Load the model from a HuggingFace checkpoint (e.g. 'KuanP/uce-cosmx-geneset').

        For backward compat, also accepts a second positional arg (token_file) which is ignored.
        """
        # Handle legacy call signature: from_pretrained(path, token_file, config=...)
        if args and isinstance(args[0], str):
            # Old API: from_pretrained(model_path, token_file, config=config)
            # Ignore token_file
            args = args[1:]

        config = kwargs.pop("config", None)
        if config is None:
            # Auto-detect from HuggingFace
            kuan_config = KuanUCEConfig.from_pretrained(pretrained_model_name_or_path)
            config = UCEConfig(
                output_embedding_dim=kuan_config.output_embedding_dim,
                d_model=kuan_config.d_model,
                nhead=kuan_config.nhead,
                num_layers=kuan_config.num_layers,
                dropout=kuan_config.dropout,
            )
        elif isinstance(config, dict):
            config = UCEConfig(**config)

        outer_model = cls(config)

        # Load Kuan's model from HuggingFace Hub
        checkpoint = get_path(["uce_paths", "checkpoint"]) if pretrained_model_name_or_path is None else pretrained_model_name_or_path
        # If checkpoint is a Path, convert to string; if it's a HF hub name, use as-is
        checkpoint = str(checkpoint)
        inner = UCEForExpressionPrediction.from_pretrained(checkpoint)
        outer_model.inner_model = inner

        logging.info(
            f"Loaded Kuan UCE model from {checkpoint}: "
            f"{sum(p.numel() for p in inner.parameters()):,} params, "
            f"output_dim={config.output_dim}"
        )

        return outer_model
