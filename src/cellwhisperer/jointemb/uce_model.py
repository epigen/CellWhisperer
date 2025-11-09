from typing import Optional
from types import SimpleNamespace

import scanpy as sc
import torch
import numpy as np
import logging
import pandas as pd
import pickle
from torch.utils.data import DataLoader
import random
import uuid

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.configuration_utils import PretrainedConfig
from transformers.processing_utils import ProcessorMixin


from cellwhisperer.config import get_path

from UCE.model import TransformerModel
from UCE.eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
from UCE.data_proc.data_utils import (
    get_species_to_pe,
    anndata_to_sc_dataset,
    data_to_torch_X,
    get_spec_chrom_csv,
    adata_path_to_prot_chrom_starts,
)

TOKEN_DIM = 5120
PE_DIM = 1280  # ESM2 embedding dimension
PAD_LENGTH = 1536
SAMPLE_SIZE = 1024
CLS_TOKEN_IDX = 3
CHROM_TOKEN_OFFSET = 143574
CHROM_TOKEN_RIGHT_IDX = 2
PAD_TOKEN_IDX = 0


class UCETranscriptomeProcessor(ProcessorMixin):
    attributes = []

    def __init__(self, nproc=8, **kwargs):
        self.name = (
            uuid.uuid4().hex
        )  # NOTE: this is not ideal but necessary to prevent file collisions... CHECK the size of the tmp files (results/UCE) and consider passing them directly in RAM
        self.nproc = nproc

        super().__init__(**kwargs)

    def _compute_features(
        self,
        adata: sc.AnnData,
        species="human",
        covar_col=np.nan,
        additional_filter=False,
    ) -> np.ndarray:
        labels = []
        if "cell_type" in adata.obs.columns:
            labels.append("cell_type")

        if covar_col is np.nan or np.isnan(covar_col):
            covar_col = None
        else:
            labels.append(covar_col)

        if additional_filter:
            sc.pp.filter_genes(adata, min_cells=10)
            sc.pp.filter_cells(adata, min_genes=25)

        _, adata = anndata_to_sc_dataset(
            adata, species=species, labels=labels, covar_col=covar_col, hv_genes=None
        )
        adata = adata.copy()

        if additional_filter:
            sc.pp.filter_genes(adata, min_cells=10)
            sc.pp.filter_cells(adata, min_genes=25)

        return adata

    def _generate_idxs(self, processed_adata):
        species_to_pe = get_species_to_pe(
            get_path(["uce_paths", "protein_embeddings_dir"])
        )
        with open(get_path(["uce_paths", "offset_pkl_path"]), "rb") as f:
            species_to_offsets = pickle.load(f)

        gene_to_chrom_pos = get_spec_chrom_csv(
            get_path(["uce_paths", "spec_chrom_csv_path"])
        )
        dataset_species = "human"
        spec_pe_genes = list(species_to_pe[dataset_species].keys())
        offset = species_to_offsets[dataset_species]
        pe_row_idxs, dataset_chroms, dataset_pos = adata_path_to_prot_chrom_starts(
            processed_adata,
            dataset_species,
            spec_pe_genes,
            gene_to_chrom_pos,
            offset,
        )

        # Save to the temp dict
        torch.save(
            {self.name: pe_row_idxs},
            get_path(["uce_paths", "tmp_pe_idx_path"], name=self.name),
        )
        with open(
            get_path(["uce_paths", "tmp_chroms_path"], name=self.name), "wb+"
        ) as f:
            pickle.dump({self.name: dataset_chroms}, f)
        with open(
            get_path(["uce_paths", "tmp_starts_path"], name=self.name), "wb+"
        ) as f:
            pickle.dump({self.name: dataset_pos}, f)

    def __call__(self, adata, *args, **kwargs) -> dict:
        """
        Preprocess and tokenize the input AnnData object for UCE model.
        Args:
            adata (AnnData): The AnnData object to process.
            *args: Additional arguments (ignored)
            **kwargs: Additional keyword arguments (ignored)
        Returns:
            dict: The processed data, as a dict of tensors.
        """

        npzs_dir = get_path(["uce_paths", "tmp_feature_path"], name=self.name)
        # Ensure the directory exists
        npzs_dir.mkdir(parents=True, exist_ok=True)
        
        processed_adata = self._compute_features(adata)
        features = data_to_torch_X(processed_adata.X).numpy()
        num_cells = processed_adata.X.shape[0]
        num_genes = processed_adata.X.shape[1]

        # Store to file (because that's how MultiDatasetSentences expects it)
        fp = np.memmap(
            npzs_dir / f"{self.name}_counts.npz",
            dtype="int64",
            mode="w+",
            shape=features.shape,
        )
        fp[:] = features[:]
        fp.flush()
        logging.info("Generated feature matrix for UCE model.")

        self._generate_idxs(processed_adata)
        logging.info("Generated idxs for UCE model.")

        # Prepare dataset and dataloader
        dataset = MultiDatasetSentences(
            sorted_dataset_names=[self.name],
            shapes_dict={self.name: (num_cells, num_genes)},
            args=SimpleNamespace(
                pad_length=PAD_LENGTH,
                sample_size=SAMPLE_SIZE,
                cls_token_idx=CLS_TOKEN_IDX,
                CHROM_TOKEN_OFFSET=CHROM_TOKEN_OFFSET,
                chrom_token_right_idx=CHROM_TOKEN_RIGHT_IDX,
                pad_token_idx=PAD_TOKEN_IDX,
            ),
            npzs_dir=str(npzs_dir) + "/",
            dataset_to_protein_embeddings_path=get_path(
                ["uce_paths", "tmp_pe_idx_path"], name=self.name
            ),
            datasets_to_chroms_path=get_path(
                ["uce_paths", "tmp_chroms_path"], name=self.name
            ),
            datasets_to_starts_path=get_path(
                ["uce_paths", "tmp_starts_path"], name=self.name
            ),
        )

        multi_dataset_sentence_collator = MultiDatasetSentenceCollator(
            pad_length=1152
        )  # hard-code here to support joint training with cellxgene_census and archs4_metasra

        dataloader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            collate_fn=multi_dataset_sentence_collator,
            num_workers=self.nproc,
        )

        data = next(iter(dataloader))
        logging.info("Generated dataset for UCE model.")

        return {
            "expression_expr": data[0],  # batch_sentences
            "expression_key_padding_mask": data[1],  # mask
        }


class UCEConfig(PretrainedConfig):
    model_type = "uce"

    def __init__(
        self,
        token_dim=TOKEN_DIM,
        d_model=PE_DIM,
        nhead=20,
        d_hid=TOKEN_DIM,
        nlayers=33,
        dropout=0.05,
        output_dim=PE_DIM,
        **kwargs,
    ):

        self.model_type = UCEConfig.model_type
        self.token_dim = token_dim
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.dropout = dropout
        self.output_dim = output_dim


class UCEModel(PreTrainedModel):
    def __init__(self, config: UCEConfig):
        super().__init__(config)
        self.config = config

        config_class = UCEConfig
        base_model_prefix = "uce_model"
        is_parallelizable = False  # not sure actually
        main_input_name = "expression_expr"  # NOTE there are actually three main inputs, but probably good enough

        self.uce_model = TransformerModel(
            d_model=config.d_model,
            nhead=config.nhead,
            d_hid=config.d_hid,
            nlayers=config.nlayers,
            dropout=config.dropout,
            output_dim=config.output_dim,
            token_dim=config.token_dim,
        )

        empty_pe = torch.zeros(145469, 5120)
        empty_pe.requires_grad = False
        self.uce_model.pe_embedding = torch.nn.Embedding.from_pretrained(empty_pe)

    def forward(
        self,
        expression_expr: torch.Tensor,
        expression_key_padding_mask: torch.Tensor,
        expression_tokens: Optional[torch.Tensor] = None,  # ignored, but needed for compatibility with other models
        expression_token_lengths: Optional[torch.Tensor] = None,  # ignored, but needed for compatibility with other models
        expression_gene: Optional[torch.Tensor] = None,  # ignored, but needed for compatibility with other models
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """
        Adapted from UCE. Vars were renamed according to our naming schema

        expression_expr = batch[0] = batch_sentences
        expression_key_padding_mask = batch[1] = mask;
        )
        """
        assert (
            expression_key_padding_mask.sum(dim=1) > 10
        ).all(), f"Num over 10: {(expression_key_padding_mask.sum(dim=1) > 10).sum()}"

        if isinstance(self.uce_model.pe_embedding, torch.nn.Embedding):
            expression_expr = expression_expr.long()  # Don't call, if captum (`InputEmbedding` wrapper class)

        expression_expr = self.uce_model.pe_embedding(expression_expr)
        expression_expr = expression_expr.permute(1, 0, 2)
        expression_expr = torch.nn.functional.normalize(
            expression_expr, dim=2
        )  # Normalize token outputs now
        return self.uce_model.forward(expression_expr, mask=expression_key_padding_mask)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path: str, token_file: str, *args, **kwargs
    ) -> PreTrainedModel:
        if "config" in kwargs:
            config = kwargs.pop("config")
            if isinstance(config, dict):
                config = UCEConfig(**config)
            elif not isinstance(config, UCEConfig):
                raise ValueError(
                    "Parameter `config` must be a dictionary or an instance of `UCEConfig`."
                )
        else:
            config = UCEConfig()
            logging.warning(
                "No configuration provided. Using default configuration from checkpoint."
            )

        outer_uce_model = cls(config, *args, **kwargs)

        # Load model state dict
        outer_uce_model.uce_model.load_state_dict(
            torch.load(pretrained_model_path, map_location="cpu"), strict=True
        )

        # Load in the protein embeddings
        all_pe = torch.load(token_file)
        if all_pe.shape[0] == 143574:
            torch.manual_seed(23)
            CHROM_TENSORS = torch.normal(mean=0, std=1, size=(1895, config.token_dim))
            # 1895 is the total number of chromosome choices, it is hardcoded for now
            all_pe = torch.vstack(
                (all_pe, CHROM_TENSORS)
            )  # Add the chrom tensors to the end
            all_pe.requires_grad = False

        if all_pe.shape[0] != 145469:
            all_pe.requires_grad = False
            outer_uce_model.uce_model.pe_embedding = torch.nn.Embedding.from_pretrained(
                all_pe
            )

        return outer_uce_model
