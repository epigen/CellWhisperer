import torch
import torch.nn as nn
from typing import Optional


class CellLevelModel(nn.Module):
    """
    Independent cell-level CNN model for processing small 56x56 cell patches.
    
    This model can operate in two modes:
    1. Standalone mode: processes cell patches independently
    2. FiLM-conditioned mode: uses context embeddings to modulate cell features via FiLM layers
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1536,
        context_dim: Optional[int] = None,
    ):
        """
        Initialize the cell-level model.
        
        Args:
            embedding_dim: Dimension of CNN feature embeddings
            num_layers: Number of CNN layers
            output_dim: Final output embedding dimension
            context_dim: Dimension of context embeddings for FiLM conditioning (None for standalone mode)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.use_film = context_dim is not None
        
        # Build CNN layers
        cnn_layers = []
        in_channels = 3
        
        for i in range(num_layers):
            if i == 0:
                # First layer: 3 -> 64 channels
                out_channels = 64
            elif i == num_layers - 1:
                # Last layer: previous -> embedding_dim
                out_channels = embedding_dim
            else:
                # Intermediate layers: double channels progressively
                out_channels = min(64 * (2 ** i), embedding_dim)
            
            cnn_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # FiLM conditioning layers (only if context_dim is provided)
        if self.use_film:
            self.film_gamma = nn.Linear(context_dim, embedding_dim)
            self.film_beta = nn.Linear(context_dim, embedding_dim)
        
        # Final projection to output dimension
        self.projection = nn.Linear(embedding_dim, output_dim)
    
    def forward(
        self,
        cell_patches: torch.Tensor,
        context_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the cell-level model.
        
        Args:
            cell_patches: Cell image patches of shape (B, 3, 56, 56)
            context_embeds: Context embeddings for FiLM conditioning (B, context_dim).
                           Required if model was initialized with context_dim.
        
        Returns:
            Cell embeddings of shape (B, output_dim)
        """
        # Validate input shape
        assert cell_patches.ndim == 4 and cell_patches.shape[1:] == (3, 56, 56), \
            f"Cell patches shape must be (B, 3, 56, 56), got {cell_patches.shape}"
        
        # Validate FiLM conditioning inputs
        if self.use_film:
            assert context_embeds is not None, \
                "context_embeds required when model uses FiLM conditioning"
            assert context_embeds.shape[1] == self.context_dim, \
                f"context_embeds must have shape (B, {self.context_dim}), got {context_embeds.shape}"
        
        # Extract CNN features
        features = self.cnn(cell_patches)  # (B, embedding_dim, H, W)
        
        # Apply FiLM conditioning if enabled
        if self.use_film and context_embeds is not None:
            gamma = self.film_gamma(context_embeds).unsqueeze(-1).unsqueeze(-1)  # (B, embedding_dim, 1, 1)
            beta = self.film_beta(context_embeds).unsqueeze(-1).unsqueeze(-1)   # (B, embedding_dim, 1, 1)
            features = features * gamma + beta
        
        # Global average pooling
        pooled = self.pool(features).squeeze(-1).squeeze(-1)  # (B, embedding_dim)
        
        # Final projection
        embeddings = self.projection(pooled)  # (B, output_dim)
        
        return embeddings
    
    @classmethod
    def create_standalone(
        cls,
        embedding_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1536,
    ) -> 'CellLevelModel':
        """
        Create a standalone cell-level model without FiLM conditioning.
        
        Args:
            embedding_dim: Dimension of CNN feature embeddings
            num_layers: Number of CNN layers
            output_dim: Final output embedding dimension
        
        Returns:
            CellLevelModel instance configured for standalone operation
        """
        return cls(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            context_dim=None,
        )
    
    @classmethod
    def create_film_conditioned(
        cls,
        context_dim: int,
        embedding_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1536,
    ) -> 'CellLevelModel':
        """
        Create a FiLM-conditioned cell-level model.
        
        Args:
            context_dim: Dimension of context embeddings for FiLM conditioning
            embedding_dim: Dimension of CNN feature embeddings
            num_layers: Number of CNN layers
            output_dim: Final output embedding dimension
        
        Returns:
            CellLevelModel instance configured for FiLM conditioning
        """
        return cls(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            context_dim=context_dim,
        )