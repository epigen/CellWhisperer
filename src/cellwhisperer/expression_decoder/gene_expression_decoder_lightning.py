import pyarrow  # needed
import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional
import logging

from cellwhisperer.jointemb.cellwhisperer_lightning import (
    TranscriptomeTextDualEncoderLightning,
)
from cellwhisperer.expression_decoder.gene_expression_decoder import (
    GeneExpressionDecoder,
    GeneExpressionDecoderConfig,
)

logger = logging.getLogger(__name__)


class GeneExpressionDecoderLightning(pl.LightningModule):
    """
    Lightning module for training gene expression decoder on frozen CellWhisperer embeddings.
    """

    def __init__(
        self,
        cellwhisperer_checkpoint: str,
        decoder_config: Dict,
        learning_rate: float = 1e-3,
        max_epochs: int = 50,
        loss_type: str = "mse",  # or "mae", "huber"
        **kwargs,
    ):
        """
        Args:
            cellwhisperer_checkpoint: Path to trained CellWhisperer checkpoint
            decoder_config: Configuration dict for the decoder
            learning_rate: Learning rate for decoder training
            max_epochs: Maximum number of training epochs
            loss_type: Loss function type (mse, mae, huber)
        """
        super().__init__()

        # Load frozen CellWhisperer model
        logger.info(f"Loading CellWhisperer checkpoint from {cellwhisperer_checkpoint}")
        self.cellwhisperer = TranscriptomeTextDualEncoderLightning.load_from_checkpoint(
            cellwhisperer_checkpoint, **kwargs
        )

        # Freeze all CellWhisperer parameters
        for param in self.cellwhisperer.parameters():
            param.requires_grad = False
        self.cellwhisperer.eval()

        # Initialize decoder
        if not isinstance(decoder_config, GeneExpressionDecoderConfig):
            decoder_config = GeneExpressionDecoderConfig(**decoder_config)

        self.decoder = GeneExpressionDecoder(decoder_config)

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.loss_type = loss_type

        self.save_hyperparameters(ignore=["cellwhisperer"])

    def forward(self, batch: Dict) -> torch.Tensor:
        """
        Forward pass: get image embeddings and predict gene expression.

        Args:
            batch: Batch dict from dataloader (must contain image data)

        Returns:
            predicted_expression: [batch_size, num_genes]
        """
        # Get image embeddings from frozen CellWhisperer
        with torch.no_grad():
            cellwhisperer_output = self.cellwhisperer(**batch)
            image_embeds = (
                cellwhisperer_output.image_embeds
            )  # [batch_size, projection_dim]

        # Predict gene expression
        predicted_expression = self.decoder(image_embeds)

        return predicted_expression

    def _compute_loss(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss between predicted and target gene expression."""
        if self.loss_type == "mse":
            return F.mse_loss(predicted, target)
        elif self.loss_type == "mae":
            return F.l1_loss(predicted, target)
        elif self.loss_type == "huber":
            return F.huber_loss(predicted, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def training_step(self, batch, batch_idx):
        """Training step."""
        # Get predicted expression
        predicted_expr = self(batch)

        # Get ground truth expression (log-transformed)
        # Assuming expression_expr contains log(counts+1) for all 6k genes
        target_expr = batch["expression_expr"]  # [batch_size, num_genes]

        # Compute loss
        loss = self._compute_loss(predicted_expr, target_expr)

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Compute correlation as additional metric
        with torch.no_grad():
            # Per-sample correlation
            correlations = []
            for pred, tgt in zip(predicted_expr, target_expr):
                corr = torch.corrcoef(torch.stack([pred, tgt]))[0, 1]
                if not torch.isnan(corr):
                    correlations.append(corr)
            if correlations:
                mean_corr = torch.stack(correlations).mean()
                self.log("train/correlation", mean_corr, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        predicted_expr = self(batch)
        target_expr = batch["expression_expr"]

        loss = self._compute_loss(predicted_expr, target_expr)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute correlation
        with torch.no_grad():
            correlations = []
            for pred, tgt in zip(predicted_expr, target_expr):
                corr = torch.corrcoef(torch.stack([pred, tgt]))[0, 1]
                if not torch.isnan(corr):
                    correlations.append(corr)
            if correlations:
                mean_corr = torch.stack(correlations).mean()
                self.log(
                    "val/correlation",
                    mean_corr,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

        return loss

    def predict_step(self, batch, batch_idx):
        """Prediction step for inference."""
        predicted_expr = self(batch)

        # Return predictions along with metadata for writing
        return {
            "predictions": predicted_expr,
            "orig_ids": batch.get("orig_ids", None),
        }

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.decoder.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=0.01,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=self.learning_rate / 20,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


from lightning.pytorch.cli import LightningCLI
from pathlib import Path
import shutil


class GeneExpressionDecoderCLI(LightningCLI):
    """CLI for gene expression decoder with custom arguments."""

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--last_model_path", type=Path, default=None)
        parser.add_argument("--best_model_path", type=Path, default=None)

    def after_fit(self) -> None:
        if self.config["fit.best_model_path"] is not None:
            # copy the best model_path
            if self.trainer.checkpoint_callback.best_model_path == "":
                logger.error(
                    "No best model path found. Please check if the checkpoint callback is enabled."
                )
            else:
                best_path = self.config["fit.best_model_path"]
                if best_path.exists():
                    logger.warning(
                        f"File {best_path} already exists. Overwriting with {self.trainer.checkpoint_callback.best_model_path}."
                    )
                shutil.copy(
                    self.trainer.checkpoint_callback.best_model_path,
                    best_path,
                )
                logger.info(f"Copied best model to {best_path}")

        if self.config["fit.last_model_path"] is not None:
            # copy the last model_path
            if self.trainer.checkpoint_callback.last_model_path == "":
                logger.error(
                    "No last model path found. Please check if the checkpoint callback is enabled."
                )
            else:
                last_path = self.config["fit.last_model_path"]
                if last_path.exists():
                    logger.warning(
                        f"File {last_path} already exists. Overwriting with {self.trainer.checkpoint_callback.last_model_path}."
                    )
                shutil.copy(
                    self.trainer.checkpoint_callback.last_model_path,
                    last_path,
                )
                logger.info(f"Copied last model to {last_path}")


def cli_main():
    """CLI entry point for training gene expression decoder."""
    import os
    from cellwhisperer.jointemb.dataset import JointEmbedDataModule
    from cellwhisperer.config import config
    from lightning.pytorch.loggers import CSVLogger

    LOG_DIR = os.path.relpath(
        config["PROJECT_ROOT"] / "results" / "expression_decoder_training", os.getcwd()
    )

    cli = GeneExpressionDecoderCLI(
        GeneExpressionDecoderLightning,
        JointEmbedDataModule,
        trainer_defaults=dict(
            default_root_dir=LOG_DIR,
            logger={
                "class_path": CSVLogger.__module__ + "." + CSVLogger.__name__,
                "init_args": dict(
                    save_dir=LOG_DIR,
                    name="decoder_logs",
                ),
            },
        ),
        save_config_callback=None,
    )


if __name__ == "__main__":
    cli_main()
