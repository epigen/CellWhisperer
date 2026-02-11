import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import anndata
from typing import Dict, Optional, Union, Tuple, Any
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.configuration_utils import PretrainedConfig
from transformers.processing_utils import ProcessorMixin
from cellwhisperer.config import get_path
from cellwhisperer.utils.processing import ensure_raw_counts_adata

logger = logging.getLogger(__name__)


class MLPTranscriptomeProcessor(ProcessorMixin):
    """Simple processor that applies log transformation to gene expression data."""

    attributes = []

    def __init__(self, gene_list_path=None, *args, **kwargs):
        # NOTE one caveat still is that the gene_list_path needs to be accessible during inference... It would be nicer if the model checkpoint "knew" which genes it was trained on and could enforce that during inference (e.g. by saving the legal gene list in the config or something)
        if gene_list_path is None:
            gene_list_path = get_path(["paths", "cosmx6k_genes"])

        # Read legal genes from CSV file (one gene per row)
        self.legal_genes = pd.read_csv(gene_list_path)["gene_name"].tolist()
        self.input_dim = len(self.legal_genes)

        logger.info(f"Loaded {self.input_dim} legal genes from {gene_list_path}")

        super().__init__(*args, **kwargs)

    def _prepare_features(self, adata):
        """
        Filter genes to legal genes in correct order and apply log transformation.
        Expects adata.X to contain raw or normalized expression counts.
        """
        # Get gene names from adata.var
        if "gene_name" in adata.var.columns:
            gene_names = adata.var["gene_name"].astype(str).str.upper()
        else:
            gene_names = adata.var.index.astype(str).str.upper()

        # Find which legal genes are present in adata
        gene_indices = []
        for legal_gene in self.legal_genes:
            matching_indices = np.where(gene_names == legal_gene)[0]
            if len(matching_indices) > 0:
                gene_indices.append(matching_indices[0])
            else:
                gene_indices.append(-1)

        # Filter to keep only present genes in correct order
        present_mask = np.array(gene_indices) != -1
        present_indices = np.array(gene_indices)[present_mask]

        logger.info(
            f"Found {len(present_indices)}/{len(self.legal_genes)} legal genes in input data"
        )

        ensure_raw_counts_adata(adata)

        # Extract data for present genes
        if hasattr(adata.X, "toarray"):
            expression_data = adata.X[:, present_indices].toarray()
        else:
            expression_data = adata.X[:, present_indices]

        # Create output array with zeros for missing genes
        full_expression = np.zeros(
            (adata.X.shape[0], len(self.legal_genes)), dtype=expression_data.dtype
        )
        full_expression[:, present_mask] = expression_data

        # Apply log transformation
        log_expression = np.log1p(full_expression)

        # Create a new AnnData object with filtered and log-transformed data
        adata_processed = anndata.AnnData(
            X=log_expression,
            obs=adata.obs.copy(),
            var=pd.DataFrame({"gene_name": self.legal_genes}, index=self.legal_genes),
        )

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
            "expression_expr": expression_data,  # Use expression_expr for consistency with other models
        }

        return batch

    @property
    def model_input_names(self):
        """Return the list of model input names that this processor produces."""
        return ["expression_expr"]


class MLPConfig(PretrainedConfig):
    """Configuration for the fully-connected MLP transcriptome model."""

    model_type = "mlp"

    def __init__(
        self,
        gene_list_path: Optional[str] = None,
        hidden_dims: list = [4096, 2048, 1024],  # does this make sense?
        output_dim: int = 1024,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Read genes from CSV to determine input_dim
        if gene_list_path is None:
            gene_list_path = str(get_path(["paths", "cosmx6k_genes"]))

        legal_genes = pd.read_csv(gene_list_path)["gene_name"].tolist()
        self.input_dim = len(legal_genes)

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
    - Input layer: input_dim (e.g. 6182 for full cosmx6k)
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
        expression_expr: torch.Tensor,
        expression_tokens: Optional[torch.Tensor] = None,  # For compatibility
        expression_token_lengths: Optional[torch.Tensor] = None,  # For compatibility
        expression_gene: Optional[torch.Tensor] = None,  # For compatibility
        expression_key_padding_mask: Optional[torch.Tensor] = None,  # For compatibility
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        """
        Forward pass through the MLP.

        Args:
            expression_expr: Log-transformed gene expression data [batch_size, input_dim]
            expression_tokens: Ignored (for compatibility with other models)
            expression_token_lengths: Ignored (for compatibility with other models)
            expression_gene: Ignored (for compatibility with other models)
            expression_key_padding_mask: Ignored (for compatibility with other models)

        Returns:
            BaseModelOutputWithPooling with last_hidden_state and pooler_output
        """
        # Pass through the MLP
        hidden_states = self.mlp(expression_expr)

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
