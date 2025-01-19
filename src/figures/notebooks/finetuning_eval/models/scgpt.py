from cellwhisperer.jointemb.scgpt_model import ScGPTConfig, ScGPTModel
import torch
import torch.nn as nn
from cellwhisperer.config import get_path
from typing import Optional


class ScGPTCelltypeModel(nn.Module):
    def __init__(self, config: ScGPTConfig, num_classes: int, freeze: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.output_layer = nn.Linear(config.hidden_size, num_classes)
        self.scgpt_model = ScGPTModel.from_pretrained(
            get_path(["model_name_path_map", "scgpt"]) / "best_model.pt", config=config
        )

        if freeze:
            for param in self.scgpt_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        expression_gene,
        expression_expr,
        expression_key_padding_mask,
        expression_tokens: torch.Tensor = None,  # ignored, needed for compatibility
        expression_token_lengths: torch.Tensor = None,  # ignored, needed for compatibility
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        (_, embs) = self.scgpt_model(
            expression_gene=expression_gene,
            expression_expr=expression_expr,
            expression_key_padding_mask=expression_key_padding_mask,
            expression_tokens=expression_tokens,  # ignored, needed for compatibility
            expression_token_lengths=expression_token_lengths,  # ignored, needed for compatibility
            return_dict=return_dict,
        )

        return self.output_layer(embs)
