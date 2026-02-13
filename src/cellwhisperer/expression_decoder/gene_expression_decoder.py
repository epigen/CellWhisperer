import torch
import torch.nn as nn
import pandas as pd
from typing import Optional
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from cellwhisperer.config import get_path

class GeneExpressionDecoderConfig(PretrainedConfig):
    """Configuration for gene expression decoder"""
    model_type = "gene_expression_decoder"
    
    def __init__(
        self,
        gene_list_path: Optional[str] = None,
        embedding_dim: int = 1024,  # CellWhisperer projection dim
        hidden_dims: list = [],  # Empty for direct linear mapping
        dropout_rate: float = 0.1,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Read genes from CSV to determine num_genes
        if gene_list_path is None:
            gene_list_path = str(get_path(["paths", "cosmx6k_genes"]))
        
        legal_genes = pd.read_csv(gene_list_path)["gene_name"].tolist()
        self.num_genes = len(legal_genes)
        
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation = activation


class GeneExpressionDecoder(PreTrainedModel):
    """
    Decoder that predicts gene expression from image embeddings.
    
    Takes image embeddings (1024-dim from CellWhisperer) and predicts
    log-transformed gene expression for all 6k genes.
    
    By default, uses a simple linear layer: embedding_dim -> num_genes
    """
    config_class = GeneExpressionDecoderConfig
    
    def __init__(self, config: GeneExpressionDecoderConfig):
        super().__init__(config)
        self.config = config
        
        layers = []
        prev_dim = config.embedding_dim
        
        # Hidden layers (if any)
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.activation.lower() == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(getattr(nn, config.activation.capitalize())())
            layers.append(nn.Dropout(config.dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (predicts log-transformed expression)
        layers.append(nn.Linear(prev_dim, config.num_genes))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            image_embeds: [batch_size, embedding_dim] image embeddings
        
        Returns:
            predicted_expression: [batch_size, num_genes] predicted log(expression+1)
        """
        return self.decoder(image_embeds)
