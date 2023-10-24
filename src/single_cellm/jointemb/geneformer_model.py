import torch
import scipy.sparse as sp
import numpy as np
import dataclasses

from typing import Optional, Union, Tuple, Any
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
        """ """

        # adata.obs["cell type"] = [x.split("#")[0] for x in adata.obs.index.values]
        # adata.obs["cell type rough"] = [
        #     x.split(".")[0] for x in adata.obs["cell type"].values
        # ]
        annot = sc.queries.biomart_annotations(
            "hsapiens",
            ["ensembl_gene_id", "external_gene_name"],
        ).set_index("external_gene_name")
        annot_drop_dups = annot.reset_index().drop_duplicates(
            subset="external_gene_name"
        )
        annot_drop_dups = annot_drop_dups.set_index("external_gene_name")

        adata_w_id = adata[:, [x for x in adata.var.index if x in annot.index]]
        adata_w_id.var["ensembl_id"] = annot_drop_dups.loc[
            adata_w_id.var.index.values, "ensembl_gene_id"
        ].values
        sc.pp.calculate_qc_metrics(adata_w_id, inplace=True)
        adata_w_id.obs["n_counts"] = adata_w_id.obs.total_counts
        adata_w_id.obs.index.name = "sample_name"
        adata_w_id.obs.reset_index(inplace=True)
        return adata_w_id

    def _tokenize(self, prepared_features, chunk_size=512, target_sum=10_000):
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
                for i in prepared_features.var["ensembl_id"][coding_miRNA_loc]
            ]
        )
        coding_miRNA_ids = prepared_features.var["ensembl_id"][coding_miRNA_loc]
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
            print(
                f"prepared_features has no column attribute 'filter_pass'; tokenizing all cells."
            )
            filter_pass_loc = np.array([i for i in range(prepared_features.shape[0])])

        tokenized_cells = []

        for i in range(0, len(filter_pass_loc), chunk_size):
            idx = filter_pass_loc[i : i + chunk_size]

            n_counts = prepared_features[idx].obs["n_counts"].values[:, None]
            X_view = prepared_features[idx, coding_miRNA_loc].X
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
        tokens, expression_token_lengths = self._tokenize(prepared_features)

        if return_tensors == "pt":
            tokens = torch.tensor(tokens, dtype=torch.long)
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
        model_directory=None,
        num_classes=0,  # TODO
        emb_mode="cell",
        hidden_size=512,
        max_ncells=200,  # TODO
        emb_layer=-1,
        emb_label=["sample_name", "cell type rough", "cell type"],  # TODO
        forward_batch_size=1,
        nproc=4,
        summary_stat=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_type = GeneformerConfig.model_type
        self.model_directory = model_directory
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
        #     "model_directory"
        # }


class GeneformerModel(PreTrainedModel):
    def __init__(
        self,
        config: GeneformerConfig,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
    ):
        super().__init__(config)
        self.config = config

        # Initialize weights and apply final processing
        # self.post_init()

        self.geneformer_model = load_model(
            "Pretrained", self.config.num_classes, self.config.model_directory
        )  # params see below

    def forward(
        self,
        expression_tokens: torch.Tensor,
        expression_token_lengths: torch.Tensor,
        # bool_masked_pos: Optional[torch.BoolTensor] = None,
        # head_mask: Optional[torch.Tensor] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # interpolate_pos_encoding: Optional[bool] = None,
        pad_token_id: Any = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        layer_to_quant = quant_layers(self.geneformer_model) + self.config.emb_layer
        embs = get_embs(
            self.geneformer_model,
            expression_tokens,
            expression_token_lengths,
            self.config.emb_mode,
            layer_to_quant,
            pad_token_id,
            self.config.forward_batch_size,
            self.config.summary_stat,
        )
        return embs

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
