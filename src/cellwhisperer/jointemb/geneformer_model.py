from cellwhisperer.config import get_path
import torch
import scipy.sparse as sp
import numpy as np
import pandas as pd
import logging
import warnings

from transformers import BertForMaskedLM, BertConfig
from typing import Dict, Optional, Union, Tuple, Any
from geneformer.in_silico_perturber import pad_tensor_list
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.configuration_utils import PretrainedConfig
from geneformer.tokenizer import TranscriptomeTokenizer, rank_genes
from transformers.processing_utils import ProcessorMixin
from pathlib import Path
from geneformer.in_silico_perturber import load_model, downsample_and_sort, quant_layers
from geneformer.emb_extractor import get_embs

import anndata
import scanpy as sc

logger = logging.getLogger(__name__)

# Set as constants here, so they are available in the TranscriptomeProcessor
PAD_TOKEN_ID = 0
MODEL_INPUT_SIZE = 2048

VERY_COMMON_GENES = {
    "FABP3",
    "FAM151A",
    "TACSTD2",
    "S100A6",
    "S100A16",
    "S100A14",
    "RGS5",
    "LEFTY1",
    "PARP1",
    "ACTA1",
    "SLC7A3",
    "BEX1",
    "CAPN6",
    "GPC4",
    "PNMA5",
    "USP9X",
    "MAGED2",
    "TSIX",
    "VGLL1",
    "TKTL1",
    "GAPDH",
}


class GeneformerTranscriptomeProcessor(ProcessorMixin):
    attributes = []

    def __init__(self, nproc, emb_label, *args, **kwargs):
        self.tokenizer = TranscriptomeTokenizer(
            custom_attr_name_dict={k: k for k in emb_label},  # refactor-delete
            nproc=nproc,
        )
        # Download
        if get_path(["paths", "ensembl_gene_symbol_map"]).exists():
            self.annot = pd.read_csv(
                get_path(["paths", "ensembl_gene_symbol_map"]), index_col=0
            )
        else:
            # Assuming gene symbol names. Use biomart to get ensembl_ids
            # use_cache=False to avoid the error sqlite3.OperationalError: database is locked
            annot = sc.queries.biomart_annotations(
                "hsapiens", ["ensembl_gene_id", "external_gene_name"], use_cache=False
            ).set_index("external_gene_name")

            annot_drop_dups = annot.reset_index().drop_duplicates(
                subset="external_gene_name"
            )
            annot_drop_dups = annot_drop_dups.set_index("external_gene_name")

            annot_drop_dups.to_csv(get_path(["paths", "ensembl_gene_symbol_map"]))
            self.annot = annot_drop_dups

        super().__init__(*args, **kwargs)

    def _prepare_features(self, adata):
        """
        Supports ensemble and symbol names
        """

        # adata.obs["cell type"] = [x.split("#")[0] for x in adata.obs.index.values]
        # adata.obs["cell type rough"] = [
        #     x.split(".")[0] for x in adata.obs["cell type"].values
        # ]
        adata_var = pd.DataFrame(adata.var)
        # no need to re-gather ensembl_id if they are already present
        if adata.var.index[0].startswith("ENSG0") or "ensembl_id" in adata.var.columns:
            # No need to translate IDs
            if "ensembl_id" in adata.var.columns:
                ensembl_ids = adata.var["ensembl_id"]
            else:
                ensembl_ids = adata.var.index
            if "." in ensembl_ids[0]:
                # Trim version
                ensembl_ids = ensembl_ids.map(
                    lambda v: v[: v.index(".") if "." in v else len(v)]
                )
            adata_var["ensembl_id"] = ensembl_ids
        else:
            if "gene_name" in adata.var.columns:
                assert (
                    len(VERY_COMMON_GENES & set(adata_var["gene_name"])) > 0
                ), f"adata.var['gene_name] should contain gene symbols but none are found. (checking these: {VERY_COMMON_GENES})"
                gene_names = adata_var["gene_name"]
            else:
                assert (
                    len(VERY_COMMON_GENES & set(adata_var.index)) > 0
                ), f"adata.var.index should contain gene symbols but none are found. (checking these: {VERY_COMMON_GENES})"
                gene_names = adata_var.index

            adata_var["ensembl_id"] = [
                self.annot["ensembl_gene_id"].get(gene_name, "")
                for gene_name in gene_names
            ]

        adata.var = adata_var

        # Filter genes that don't have an ensembl_id (according to our mapping file)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*is_categorical_dtype is deprecated.*"
            )
            adata = adata[:, [x.startswith("ENSG0") for x in adata.var["ensembl_id"]]]
            # alternatively check for `x in annot["ensembl_gene_id"].values`

        # Since the implicit copy() mechanism seems to be broken, I need to do it explicitly
        adata_w_id = anndata.AnnData(
            X=adata.X.copy(),
            var=pd.DataFrame(adata.var),
            obs=pd.DataFrame(adata.obs),
        )

        # if isinstance(adata_w_id.X, anndata._core.views.ArrayView):  # use this code snippets, if complications arise with the copy() above
        #     X = np.array(adata_w_id.X)
        # elif isinstance(adata_w_id.X, anndata._core.views.SparseCSRView):
        #     X = adata_w_id.X.copy()

        sc.pp.calculate_qc_metrics(adata_w_id, inplace=True)
        adata_w_id.obs["n_counts"] = adata_w_id.obs.total_counts
        if "sample_name" in adata_w_id.obs.columns:
            # rename column
            adata_w_id.obs.rename(
                columns={"sample_name": "sample_name_attrib"}, inplace=True
            )

        adata_w_id.obs.index.name = "sample_name"
        adata_w_id.obs.reset_index(inplace=True)
        return adata_w_id

    def _tokenize(
        self, prepared_features, chunk_size=512, target_sum=10_000, padding=False
    ):
        """
        Args:
            prepared_features: anndata object

        """
        # copied from def tokenize_anndata(self, adata_file_path, target_sum=10_000, chunk_size=512):

        coding_and_miRNA_loc = np.where(
            [
                self.tokenizer.genelist_dict.get(i, False)
                for i in prepared_features.var["ensembl_id"]
            ]
        )[0]
        norm_factor_vector = np.array(
            [
                self.tokenizer.gene_median_dict[i]
                for i in prepared_features.var["ensembl_id"].iloc[coding_and_miRNA_loc]
            ]
        )
        coding_miRNA_ids = prepared_features.var["ensembl_id"].iloc[
            coding_and_miRNA_loc
        ]
        coding_miRNA_tokens = np.array(
            [self.tokenizer.gene_token_dict[i] for i in coding_miRNA_ids]
        )

        try:
            filter_pass_loc = np.where(
                [i == 1 for i in prepared_features.obs["filter_pass"]]
            )[0]
        except KeyError:
            logger.debug(
                "prepared_features has no column attribute 'filter_pass'; tokenizing all cells."
            )
            filter_pass_loc = np.array([i for i in range(prepared_features.shape[0])])

        tokenized_cells = []

        for i in range(0, len(filter_pass_loc), chunk_size):
            idx = filter_pass_loc[i : i + chunk_size]

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*is_categorical_dtype is deprecated.*"
                )
                n_counts = prepared_features[idx].obs["n_counts"].values[:, None]
                X_view = prepared_features[idx, coding_and_miRNA_loc].X

            X_norm = X_view / n_counts * target_sum / norm_factor_vector

            X_norm = sp.csr_matrix(X_norm)

            tokenized_cells += [
                rank_genes(X_norm[i].data, coding_miRNA_tokens[X_norm[i].indices])
                for i in range(X_norm.shape[0])
            ]

        # truncate dataset (this is adapted from geneformer)
        # NOTE: strong limitation of 2048 tokens -.-
        tokenized_cells = [tc[:MODEL_INPUT_SIZE] for tc in tokenized_cells]
        expression_token_lengths = [len(tc) for tc in tokenized_cells]

        if not all(expression_token_lengths):
            logger.warning(
                f"Encountered {(np.array(expression_token_lengths) == 0).sum()} cell(s) with 0 counts or gene with 0 median expression. Hacking low expression here to prevent nans and subsequent failures"
            )
            expression_token_lengths = [
                1 if l == 0 else l for l in expression_token_lengths
            ]
            # I picked `9743` randomly. It should anyways be filtered
            tokenized_cells = [
                np.array([9743], dtype=np.int16) if len(tc) == 0 else tc
                for tc in tokenized_cells
            ]

        return tokenized_cells, expression_token_lengths

    def __call__(
        self, features, return_tensors=None, *args, **kwargs
    ) -> Dict[str, Any]:
        prepared_features = self._prepare_features(features)
        tokens, expression_token_lengths = self._tokenize(
            prepared_features, *args, **kwargs
        )
        assert all(expression_token_lengths), "Some cells have 0 genes"

        if return_tensors == "pt":
            max_len = max(expression_token_lengths)
            tokens = [torch.from_numpy(v).to(dtype=torch.long) for v in tokens]
            tokens = pad_tensor_list(
                tokens,
                max_len,
                pad_token_id=PAD_TOKEN_ID,
                model_input_size=MODEL_INPUT_SIZE,
            )
            expression_token_lengths = torch.tensor(expression_token_lengths)
        elif return_tensors is not None:
            raise ValueError("return_tensors must be 'pt' (PyTorch) or None!")

        return {
            "expression_tokens": tokens,
            "expression_token_lengths": expression_token_lengths,
        }

    @property
    def model_input_names(self):
        return ["expression_tokens", "expression_token_lengths"]


class GeneformerConfig(PretrainedConfig):
    model_type = "geneformer"

    def __init__(
        self,
        num_classes=0,  # refactor-delete
        emb_mode="cell",
        hidden_size=512,
        max_ncells=200,  # refactor-delete
        emb_layer=-1,
        emb_label=["sample_name", "cell type rough", "cell type"],  # refactor-delete
        forward_batch_size=-1,
        nproc=4,
        summary_stat=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_type = GeneformerConfig.model_type
        self.num_classes = num_classes
        self.emb_mode = emb_mode
        self.hidden_size = hidden_size
        self.max_ncells = max_ncells
        self.emb_layer = emb_layer
        self.emb_label = emb_label
        self.forward_batch_size = forward_batch_size
        self.nproc = nproc
        self.summary_stat = summary_stat

        # valid_option_dict = {
        #     "num_classes": {int},
        #     "emb_mode": {"cell", "gene"},
        #     "cell_emb_style": {"mean_pool"},
        #     "filter_data": {None, dict},
        #     "max_ncells": {None, int},
        #     "emb_layer": {-1, 0},
        #     "emb_label": {None, list},
        #     "forward_batch_size": {int},
        #     "nproc": {int},
        #     "summary_stat": {None, "mean", "median"},
        # }


class GeneformerModel(
    PreTrainedModel
):  # we could even subclass from BertForMaskedLM. but how to then do the config thing?
    config_class = GeneformerConfig
    base_model_prefix = "geneformer_model"
    is_parallelizable = False  # not sure actually
    main_input_name = "expression_tokens"

    def __init__(self, config: GeneformerConfig):
        super().__init__(config)
        self.config = config

        # model configuration
        bert_config = {
            "hidden_size": 512,
            "num_hidden_layers": 12,
            "initializer_range": 0.2,
            "layer_norm_eps": 1e-12,
            "attention_probs_dropout_prob": 0.02,
            "hidden_dropout_prob": 0.02,
            "intermediate_size": 1024,
            "hidden_act": "relu",
            "max_position_embeddings": 2**11,
            "model_type": "bert",
            "num_attention_heads": 4,
            "pad_token_id": PAD_TOKEN_ID,
            "output_hidden_states": True,
            "output_attentions": False,
            "vocab_size": 25426,  # genes+2 for <mask> and <pad> tokens
        }

        self.geneformer_model = BertForMaskedLM(BertConfig(**bert_config))
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        expression_tokens: torch.Tensor,
        expression_token_lengths: torch.Tensor,
        expression_gene=None,  # ignored, but needed for compatibility with other models
        expression_expr=None,  # ignored, but needed for compatibility with other models
        expression_key_padding_mask=None,  # ignored, but needed for compatibility with other models
        # bool_masked_pos: Optional[torch.BoolTensor] = None,
        # head_mask: Optional[torch.Tensor] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        assert not return_dict, f"No support for return_dict={return_dict}"
        layer_to_quant = quant_layers(self.geneformer_model) + self.config.emb_layer
        embs = get_embs(
            self.geneformer_model,
            expression_tokens,
            expression_token_lengths,
            self.config.emb_mode,
            layer_to_quant,
            PAD_TOKEN_ID,
            self.config.forward_batch_size,
            self.config.summary_stat,
        )
        return (None, embs)

        # if not return_dict:
        #     head_outputs = (
        #         (sequence_output, pooled_output)
        #         if pooled_output is not None
        #         else (sequence_output,)
        #     )
        #     return head_outputs + encoder_outputs[1:]

        # return BaseModelOutputWithPooling(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        # )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ) -> PreTrainedModel:
        if "config" in kwargs:
            config = kwargs.pop("config")
            if isinstance(config, dict):
                config = GeneformerConfig(**config)
            elif not isinstance(config, GeneformerConfig):
                raise ValueError(
                    "Parameter `config` must be a dictionary or an instance of `GeneformerConfig`."
                )
        else:
            config = GeneformerConfig()
            logger.warning(
                "No configuration provided. Using default configuration from checkpoint."
            )

        model = cls(config, *args, **kwargs)

        model.geneformer_model = BertForMaskedLM.from_pretrained(
            pretrained_model_name_or_path,
            output_hidden_states=True,
            output_attentions=False,
        )
        return model
