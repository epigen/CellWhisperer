"""
Custom Lightning callback to write gene expression predictions to h5ad file.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence

import anndata
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import BasePredictionWriter

from cellwhisperer.config import get_path

logger = logging.getLogger(__name__)


class GeneExpressionPredictionWriter(BasePredictionWriter):
    """
    Callback to write gene expression predictions to h5ad file.
    
    This callback collects predictions from all batches and writes them
    to an AnnData object with metadata from the original dataset.
    """

    def __init__(
        self,
        output_path: str,
        dataset_name: str,
        write_interval: str = "epoch",
    ):
        """
        Args:
            output_path: Path to save the h5ad file with predictions
            dataset_name: Name of the dataset being predicted on
            write_interval: When to write predictions ('batch', 'epoch', 'batch_and_epoch')
        """
        super().__init__(write_interval)
        self.output_path = Path(output_path)
        self.dataset_name = dataset_name
        self.predictions = []
        self.orig_ids = []

    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ):
        """Write predictions at the end of the epoch."""
        logger.info("Collecting predictions...")
        
        # Collect all predictions and metadata
        all_predictions = []
        all_orig_ids = []
        
        for batch_preds in predictions:
            for pred_dict in batch_preds:
                all_predictions.append(pred_dict["predictions"].cpu().numpy())
                if pred_dict["orig_ids"] is not None:
                    all_orig_ids.extend(pred_dict["orig_ids"])
        
        # Concatenate predictions
        predictions_array = np.concatenate(all_predictions, axis=0)
        logger.info(f"Predicted expression shape: {predictions_array.shape}")
        
        # Load original dataset for metadata
        logger.info(f"Loading original dataset: {self.dataset_name}")
        dataset_path = get_path(["paths", "full_dataset"], dataset=self.dataset_name)
        
        # Check if it's a multi-file dataset
        if not dataset_path.exists():
            # Try multi-file format
            tma_id = self.dataset_name.split("_")[-1]
            dataset_path = get_path(
                ["paths", "full_dataset_multi"],
                dataset="lymphoma_cosmx_large",
                i=tma_id,
            )
        
        logger.info(f"Loading from: {dataset_path}")
        original_adata = anndata.read_h5ad(dataset_path)
        
        # Get gene names
        gene_list_path = get_path(["paths", "cosmx6k_genes"])
        gene_df = pd.read_csv(gene_list_path)
        gene_names = gene_df["gene_name"].tolist()
        
        logger.info(f"Number of genes: {len(gene_names)}")
        
        # Match order with original dataset if orig_ids available
        if all_orig_ids:
            original_adata = original_adata[all_orig_ids]
        
        # Create AnnData with predicted expression
        predicted_adata = anndata.AnnData(
            X=predictions_array,
            obs=original_adata.obs.copy(),
            var=pd.DataFrame({"gene_name": gene_names}, index=gene_names),
        )
        
        # Copy spatial coordinates if available
        if "spatial" in original_adata.obsm:
            predicted_adata.obsm["spatial"] = original_adata.obsm["spatial"].copy()
        
        # Store ground truth expression in .layers for comparison
        common_genes = list(set(gene_names) & set(original_adata.var_names))
        logger.info(
            f"Common genes between prediction and ground truth: {len(common_genes)}"
        )
        
        if common_genes:
            gt_indices = [gene_names.index(g) for g in common_genes]
            orig_indices = [
                list(original_adata.var_names).index(g) for g in common_genes
            ]
            
            ground_truth_full = np.zeros_like(predictions_array)
            ground_truth_full[:, gt_indices] = (
                original_adata.X[:, orig_indices].toarray()
                if hasattr(original_adata.X, "toarray")
                else original_adata.X[:, orig_indices]
            )
            
            predicted_adata.layers["ground_truth"] = ground_truth_full
            predicted_adata.uns["common_genes"] = common_genes
        
        # Add metadata
        predicted_adata.uns["dataset_name"] = self.dataset_name
        predicted_adata.uns["prediction_type"] = "gene_expression_decoder"
        
        # Save predictions
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving predictions to: {self.output_path}")
        predicted_adata.write_h5ad(self.output_path)
        
        logger.info("Prediction complete!")
        logger.info(f"Output saved to: {self.output_path}")
        logger.info(f"Shape: {predicted_adata.shape}")
