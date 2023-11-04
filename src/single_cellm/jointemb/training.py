#!/usr/bin/env python
# coding: utf-8
"""
See https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html for documentation on usage
"""

import torch
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
import subprocess
from pathlib import Path
import os
import yaml
import logging
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import Logger, WandbLogger
from single_cellm.jointemb.single_cellm_lightning import (
    TranscriptomeTextDualEncoderLightning,
)
from single_cellm.jointemb.dataset import JointEmbedDataModule


PROJECT_DIR = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=os.getcwd())
    .decode("utf-8")
    .strip()
)


class SingleCeLLMCLI(LightningCLI):
    """CLI for single-cellm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--freeze_pretrained", default=True)

    def before_fit(self) -> None:
        # We need to preload this model
        try:
            self.model.load_pretrained_models(
                PROJECT_DIR / "resources" / "geneformer-12L-30M"
            )
        except FileNotFoundError:
            logging.error(
                "Unabld to fine geneformer model. Please download first (see `rna` snakemake pipeline)"
            )
            raise
        if self.config["fit.freeze_pretrained"]:
            self.model.freeze_pretrained()


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(
                self.config, skip_none=False
            )  # Required for proper reproducibility
            trainer.logger.log_hyperparams({"config": config})


def cli_main():
    torch.set_float32_matmul_precision("high")  # speed up on ampere-level GPUs
    torch.multiprocessing.set_sharing_strategy("file_system")

    with open(PROJECT_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)
        LOG_DIR = PROJECT_DIR / config["paths"]["wandb_logs"]

    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=1e-5, patience=20, verbose=False, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        save_last=True,
        filename="{epoch}-{val_loss:.2f}",
    )

    SingleCeLLMCLI(
        TranscriptomeTextDualEncoderLightning,
        JointEmbedDataModule,
        trainer_defaults=dict(
            # accelerator="gpu",
            default_root_dir=LOG_DIR,
            logger={
                "class_path": WandbLogger.__module__ + "." + WandbLogger.__name__,
                "init_args": dict(
                    save_dir=LOG_DIR,
                    project="JointEmbed_Training",
                    entity="single-cellm",
                    log_model=False,
                ),
            },
            enable_progress_bar=True,
            callbacks=[checkpoint_callback, early_stop],
        ),
        save_config_callback=None,
    )


if __name__ == "__main__":
    cli_main()
