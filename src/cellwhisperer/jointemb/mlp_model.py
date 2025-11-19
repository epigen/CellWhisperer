import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Union, Tuple, Any
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.configuration_utils import PretrainedConfig
from transformers.processing_utils import ProcessorMixin

logger = logging.getLogger(__name__)


class MLPTranscriptomeProcessor(ProcessorMixin):
    """Simple processor that applies log transformation to gene expression data."""

    attributes = []

    def __init__(self, input_dim=1000, *args, **kwargs):
        self.input_dim = input_dim
        super().__init__(*args, **kwargs)

    def _prepare_features(self, adata):
        """
        Apply log transformation to expression data.
        Expects adata.X to contain raw or normalized expression counts.
        """
        # Ensure we have the correct number of genes
        if adata.X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} genes, but got {adata.X.shape[1]} genes. "
                f"Please filter the data to exactly {self.input_dim} genes."
            )

        # Apply log transformation: log(x + 1) to handle zeros
        if hasattr(adata.X, "toarray"):
            # Handle sparse matrices
            expression_data = adata.X.toarray()
        else:
            # Handle dense matrices
            expression_data = adata.X

        # Apply log transformation
        log_expression = np.log(expression_data + 1)

        # Create a copy of adata with log-transformed data
        adata_processed = adata.copy()
        adata_processed.X = log_expression

        return adata_processed

    def __call__(self, adata, **kwargs):
        """
        Process the adata and return features in the expected format.
        """
        adata_processed = self._prepare_features(adata)

        # Convert to tensors
        expression_data = torch.FloatTensor(adata_processed.X)

        # Create batch with consistent naming for compatibility
        batch = {
            "expression_data": expression_data,
            # Add dummy values for compatibility with existing interface
            "expression_tokens": torch.zeros(
                (expression_data.shape[0], 1), dtype=torch.long
            ),
            "expression_token_lengths": torch.ones(
                expression_data.shape[0], dtype=torch.long
            ),
        }

        return batch


class MLPConfig(PretrainedConfig):
    """Configuration for the fully-connected MLP transcriptome model."""

    model_type = "mlp"

    def __init__(
        self,
        input_dim: int = 1000,
        hidden_dims: list = [512, 256, 512, 1024],  # does this make sense?
        output_dim: int = 1024,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.activation = activation

        if self.activation.lower() != "relu":
            logger.warning(f"Using {self.activation} activation instead of ReLU")


class MLPModel(PreTrainedModel):
    """
    Fully-connected neural network for transcriptome data.

    Architecture:
    - Input layer: input_dim (1000)
    - 4 hidden layers with ReLU activation
    - Output layer: output_dim
    - Dropout between layers
    """

    config_class = MLPConfig

    def __init__(self, config: MLPConfig):
        super().__init__(config)
        self.config = config

        # Build the network layers
        layers = []
        prev_dim = config.input_dim

        # Add hidden layers
        for i, hidden_dim in enumerate(config.hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Activation
            if config.activation.lower() == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(getattr(nn, config.activation.capitalize())())

            # Dropout (except after the last hidden layer)
            if i < len(config.hidden_dims) - 1:
                layers.append(nn.Dropout(config.dropout_rate))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, config.output_dim))

        # Create the sequential model
        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        expression_data: torch.Tensor,
        expression_tokens: Optional[torch.Tensor] = None,  # For compatibility
        expression_token_lengths: Optional[torch.Tensor] = None,  # For compatibility
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        """
        Forward pass through the MLP.

        Args:
            expression_data: Log-transformed gene expression data [batch_size, input_dim]
            expression_tokens: Ignored (for compatibility with other models)
            expression_token_lengths: Ignored (for compatibility with other models)

        Returns:
            BaseModelOutputWithPooling with last_hidden_state and pooler_output
        """
        # Pass through the MLP
        hidden_states = self.mlp(expression_data)

        # For compatibility with the existing interface, we need to provide both
        # last_hidden_state and pooler_output
        # Since this is a simple MLP, we use the same output for both

        # Add sequence dimension for compatibility (batch_size, seq_len=1, hidden_dim)
        last_hidden_state = hidden_states.unsqueeze(1)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=hidden_states,  # This will be used for downstream tasks
            hidden_states=None,
            attentions=None,
        )

    def get_input_embeddings(self):
        """Return the first linear layer as input embeddings (for compatibility)."""
        return self.mlp[0]

    def set_input_embeddings(self, value):
        """Set the first linear layer (for compatibility)."""
        self.mlp[0] = value
