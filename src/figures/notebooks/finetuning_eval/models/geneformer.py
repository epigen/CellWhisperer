from cellwhisperer.jointemb.geneformer_model import GeneformerModel, GeneformerConfig
import torch
import torch.nn as nn
from cellwhisperer.config import get_path
from typing import Optional


class GeneformerCelltypeModel(nn.Module):
    def __init__(self, config: GeneformerConfig, num_classes: int, freeze: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.output_layer = nn.Linear(config.hidden_size, num_classes)
        self.geneformer_model = GeneformerModel.from_pretrained(
            get_path(["model_name_path_map", "geneformer"]), config=config
        )

        if freeze:
            for param in self.geneformer_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        expression_tokens: torch.Tensor,
        expression_token_lengths: torch.Tensor,
        expression_gene=None,  # ignored, but needed for compatibility with other models
        expression_expr=None,  # ignored, but needed for compatibility with other models
        expression_key_padding_mask=None,  # ignored, but needed for compatibility with other models
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        (_, embs) = self.geneformer_model(
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            expression_gene=expression_gene,
            expression_expr=expression_expr,
            expression_key_padding_mask=expression_key_padding_mask,
            return_dict=return_dict,
        )

        return self.output_layer(embs)
