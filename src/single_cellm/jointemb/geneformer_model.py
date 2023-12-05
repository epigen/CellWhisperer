from single_cellm.config import get_path
import torch
import scipy.sparse as sp
import numpy as np
import pandas as pd
import logging
import warnings

from transformers import BertForMaskedLM, BertConfig
from typing import Optional, Union, Tuple, Any
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


class GeneformerTranscriptomeProcessor(ProcessorMixin):
    """
    TODO: move the feauture processing and tokenization (in the model below) here
    """

    attributes = []

    def __init__(self, nproc, emb_label, *args, **kwargs):
        self.tokenizer = TranscriptomeTokenizer(
            custom_attr_name_dict={k: k for k in emb_label},
            nproc=nproc,
        )
        super().__init__(*args, **kwargs)

    def _prepare_features(self, adata):
        """
        Supports ensemble and symbol names
        """

        # adata.obs["cell type"] = [x.split("#")[0] for x in adata.obs.index.values]
        # adata.obs["cell type rough"] = [
        #     x.split(".")[0] for x in adata.obs["cell type"].values
        # ]
        if adata.var.index[0].startswith("ENSG0"):
            # No need to translate IDs
            ensembl_ids = adata.var.index
            if "." in ensembl_ids[0]:
                # Trim version
                ensembl_ids = ensembl_ids.map(lambda v: v[: v.index(".")])
            adata_w_id = adata
            adata_var = pd.DataFrame(adata_w_id.var)
            adata_var["ensembl_id"] = ensembl_ids
            adata_w_id = anndata.AnnData(
                X=adata_w_id.X,
                var=adata_var,
                obs=pd.DataFrame(adata_w_id.obs),
            )
        else:
            annot = pd.read_csv(
                get_path(["paths", "ensembl_gene_symbol_map"]), index_col=0
            )
            # TODO add assertion that adata.var.index contains the gene names

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*is_categorical_dtype is deprecated.*"
                )
                adata_w_id = adata[:, [x for x in adata.var.index if x in annot.index]]
            # Since the implicit copy() mechanism seems to be broken, I need to do it explicitly
            adata_w_id = anndata.AnnData(
                X=adata_w_id.X.copy(),
                var=pd.DataFrame(adata_w_id.var),
                obs=pd.DataFrame(adata_w_id.obs),
            )

            # if isinstance(adata_w_id.X, anndata._core.views.ArrayView):  # use this code snippets, if complications arise with the copy() above
            #     X = np.array(adata_w_id.X)
            # elif isinstance(adata_w_id.X, anndata._core.views.SparseCSRView):
            #     X = adata_w_id.X.copy()

            adata_w_id.var["ensembl_id"] = annot.loc[
                adata_w_id.var.index.values, "ensembl_gene_id"
            ].values

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

        coding_miRNA_loc = np.where(
            [
                self.tokenizer.genelist_dict.get(i, False)
                for i in prepared_features.var["ensembl_id"]
            ]
        )[0]
        norm_factor_vector = np.array(
            [
                self.tokenizer.gene_median_dict[i]
                for i in prepared_features.var["ensembl_id"].iloc[coding_miRNA_loc]
            ]
        )
        coding_miRNA_ids = prepared_features.var["ensembl_id"].iloc[coding_miRNA_loc]
        coding_miRNA_tokens = np.array(
            [self.tokenizer.gene_token_dict[i] for i in coding_miRNA_ids]
        )

        try:
            _ = prepared_features.obs["filter_pass"]
        except KeyError:
            var_exists = False
        else:
            var_exists = True

        if var_exists:
            filter_pass_loc = np.where(
                [i == 1 for i in prepared_features.obs["filter_pass"]]
            )[0]
        else:
            logging.info(
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
                X_view = prepared_features[idx, coding_miRNA_loc].X
            with warnings.catch_warnings():  # We can ignore this warning, because we later filter for cells with 0 counts
                warnings.filterwarnings(
                    "ignore", message=".*divide by zero encountered in divide.*"
                )
                X_norm = X_view / n_counts * target_sum / norm_factor_vector
            X_norm = sp.csr_matrix(X_norm)

            tokenized_cells += [
                rank_genes(X_norm[i].data, coding_miRNA_tokens[X_norm[i].indices])
                for i in range(X_norm.shape[0])
            ]

        # truncate dataset (this is adapted from geneformer)
        # TODO strong limitation of 2048 tokens -.-
        tokenized_cells = [tc[:2048] for tc in tokenized_cells]
        expression_token_lengths = [len(tc) for tc in tokenized_cells]

        return tokenized_cells, expression_token_lengths

    def __call__(self, features, return_tensors=None, *args, **kwargs):
        prepared_features = self._prepare_features(features)
        tokens, expression_token_lengths = self._tokenize(
            prepared_features, *args, **kwargs
        )

        if return_tensors == "pt":
            max_len = max(expression_token_lengths)
            tokens = [torch.from_numpy(v).to(dtype=torch.long) for v in tokens]
            tokens = pad_tensor_list(
                tokens, max_len, pad_token_id=0, model_input_size=2048
            )  # TODO get pad_token_id and model_size from config
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
        num_classes=0,  # TODO
        emb_mode="cell",
        hidden_size=512,
        max_ncells=200,  # TODO
        emb_layer=-1,
        emb_label=["sample_name", "cell type rough", "cell type"],  # TODO
        forward_batch_size=-1,
        nproc=4,
        summary_stat=None,
        pad_token_id=0,
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
        self.pad_token_id = pad_token_id

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

    def __init__(
        self,
        config: GeneformerConfig,
        add_pooling_layer: bool = True,  # TODO refactor-delete
        use_mask_token: bool = False,  # TODO refactor-delete
    ):
        super().__init__(config)
        self.config = config

        # model configuration
        bert_config = {
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "initializer_range": 0.2,
            "layer_norm_eps": 1e-12,
            "attention_probs_dropout_prob": 0.02,
            "hidden_dropout_prob": 0.02,
            "intermediate_size": 1024,
            "hidden_act": "relu",
            "max_position_embeddings": 2**11,
            "model_type": "bert",
            "num_attention_heads": 4,
            "pad_token_id": 0,
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
            self.config.pad_token_id,
            self.config.forward_batch_size,
            self.config.summary_stat,
        )
        return (embs,)

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
            logging.warning(
                "No configuration provided. Using default configuration from checkpoint."
            )

        model = cls(config, *args, **kwargs)

        model.geneformer_model = BertForMaskedLM.from_pretrained(
            pretrained_model_name_or_path,
            output_hidden_states=True,
            output_attentions=False,
        )
        return model
