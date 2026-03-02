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
from torch.utils.data import ConcatDataset

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
        
        # Collect all predictions
        all_predictions = []
        for pred_dict in predictions:
            preds = pred_dict["predictions"].cpu().numpy()
            all_predictions.append(preds)
        
        # Concatenate predictions
        predictions_array = np.concatenate(all_predictions, axis=0)
        logger.info(f"Predicted expression shape: {predictions_array.shape}")
        
        # Get orig_ids from the dataset (not from batches, as they're not included with disk loading)
        # orig_ids is metadata stored at the dataset level, not in individual samples
        predict_dataloaders = trainer.predict_dataloaders
        
        if predict_dataloaders is None:
            raise RuntimeError("No predict dataloaders available from trainer")
        
        # Handle both single dataloader and list of dataloaders
        if isinstance(predict_dataloaders, list):
            if len(predict_dataloaders) == 0:
                raise RuntimeError("No predict dataloaders available from trainer")
            if len(predict_dataloaders) > 1:
                logger.warning(
                    f"Multiple predict dataloaders detected ({len(predict_dataloaders)}). "
                    "Using only the first dataloader for orig_ids extraction."
                )
            dataloader = predict_dataloaders[0]
        else:
            # Single dataloader (not a list)
            dataloader = predict_dataloaders
        
        dataset = dataloader.dataset
        
        # Handle ConcatDataset (multiple datasets concatenated)
        if isinstance(dataset, ConcatDataset):
            all_orig_ids = []
            for ds in dataset.datasets:
                ds_orig_ids = ds.orig_ids
                # Convert to list if it's a numpy array
                if hasattr(ds_orig_ids, 'tolist'):
                    all_orig_ids.extend(ds_orig_ids.tolist())
                else:
                    all_orig_ids.extend(ds_orig_ids)
        else:
            # Single dataset
            ds_orig_ids = dataset.orig_ids
            if hasattr(ds_orig_ids, 'tolist'):
                all_orig_ids = ds_orig_ids.tolist()
            else:
                all_orig_ids = list(ds_orig_ids)
        
        logger.info(f"Number of orig_ids from dataset: {len(all_orig_ids)}")
        
        # Warn if predictions count doesn't match dataset length
        if len(all_orig_ids) != predictions_array.shape[0]:
            logger.warning(
                f"Mismatch detected: dataset has {len(all_orig_ids)} orig_ids, "
                f"but received {predictions_array.shape[0]} predictions. "
                "This could indicate dropped samples during prediction."
            )
        
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
        
        # Match order with original dataset using orig_ids
        # orig_ids must be available from the dataset - fail loudly if not
        assert all_orig_ids, "orig_ids must be available from dataset"
        
        logger.info(f"Filtering original dataset using {len(all_orig_ids)} orig_ids")
        original_adata = original_adata[all_orig_ids]
        
        # Ensure dimensions match after filtering
        assert original_adata.n_obs == predictions_array.shape[0], (
            f"Mismatch after filtering: original_adata has {original_adata.n_obs} obs, "
            f"but predictions have {predictions_array.shape[0]} rows"
        )
        
        # Create AnnData with predicted expression
        predicted_adata = anndata.AnnData(
            X=predictions_array,
            obs=original_adata.obs.copy(),
            var=pd.DataFrame({"gene_name": gene_names}, index=gene_names),
        )
        
        # Copy spatial coordinates if available
        if "spatial" in original_adata.obsm:
            predicted_adata.obsm["spatial"] = original_adata.obsm["spatial"][:predictions_array.shape[0]].copy()
        
        # Store ground truth expression in .layers for comparison
        common_genes = list(set(gene_names) & set(original_adata.var_names))
        logger.info(
            f"Common genes between prediction and ground truth: {len(common_genes)}"
        )
        
        assert common_genes, "No common genes found between predictions and ground truth"
        
        gt_indices = [gene_names.index(g) for g in common_genes]
        orig_indices = [
            list(original_adata.var_names).index(g) for g in common_genes
        ]
        
        ground_truth_full = np.zeros_like(predictions_array)
        ground_truth_subset = (
            original_adata.X[:predictions_array.shape[0], orig_indices].toarray()
            if hasattr(original_adata.X, "toarray")
            else original_adata.X[:predictions_array.shape[0], orig_indices]
        )
        # Apply log1p to match the decoder's output space.
        # The decoder is trained on log1p(counts) (applied by MLPTranscriptomeProcessor),
        # so ground truth must also be in log1p space for meaningful comparison.
        ground_truth_full[:, gt_indices] = np.log1p(ground_truth_subset)
        
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
