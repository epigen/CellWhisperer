#!/usr/bin/env python
# coding: utf-8
"""
See https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html for documentation on usage
"""
from lightning.pytorch.tuner import Tuner
from single_cellm.config import get_path, model_path_from_name
from single_cellm.misc.utils import obj_signature
import torch
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
import subprocess
import shutil
from pathlib import Path
import os
import yaml
import logging
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import Logger, WandbLogger
from jsonargparse import lazy_instance

from single_cellm.jointemb.model import TranscriptomeTextDualEncoderConfig
from single_cellm.jointemb.loss.config import LossConfig
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
        parser.add_argument("--best_model_path", type=Path, default=None)
        parser.add_argument("--log_level", default="WARNING")
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument(
            "--wandb",
            default="",
            help="Enable WandB logging. Need to pass a run name",
            type=str,
        )
        parser.set_defaults(
            {
                "model.model_config": lazy_instance(
                    TranscriptomeTextDualEncoderConfig,
                    **obj_signature(TranscriptomeTextDualEncoderConfig),
                ),
                "model.loss_config": lazy_instance(
                    LossConfig, **obj_signature(LossConfig)
                ),
                "trainer.max_epochs": 100,
            }
        )

        parser.link_arguments(
            "trainer.max_epochs", "model.max_epochs"
        )  # needed for scheduler initialization
        parser.link_arguments("wandb", "trainer.logger.init_args.name")
        parser.link_arguments(
            "wandb",
            "trainer.logger.init_args.mode",
            lambda wandb: "disabled" if wandb == "" else "online",
        )
        # TODO not working, because model_type needs to be top level within model_config
        # parser.link_arguments(
        #     "model.model_config.transcriptome_config.model_type",
        #     "data.transcriptome_processor",
        # )
        # parser.link_arguments(
        #     "model.model_config.text_config.model_type", "data.tokenizer"
        # )

        parser.link_arguments(
            ["trainer.fast_dev_run", "batch_size"],
            "data.batch_size",
            compute_fn=lambda fast_dev_run, batch_size: 2  # TODO 1 fails maybe :/
            if fast_dev_run
            else batch_size,
        )

    def before_instantiate_classes(self) -> None:
        logging.basicConfig(level=self.config["fit.log_level"])

    def before_fit(self) -> None:
        # We need to preload this model
        try:
            self.model.load_pretrained_models(
                model_path_from_name(
                    self.model.model.transcriptome_model.config.model_type
                ),
                model_path_from_name(self.model.model.text_model.config.model_type),
            )
        except FileNotFoundError:
            logging.error(
                "Unable to find geneformer model. Please download first (see `rna` snakemake pipeline)"
            )
            raise

        if not self.datamodule.batch_size > 0:
            tuner = Tuner(self.trainer)
            tuner.scale_batch_size(
                self.model, datamodule=self.datamodule, mode="binsearch", init_val=8
            )
            
        # self.trainer.logger.watch(self.model, log="all")

    def after_fit(self) -> None:
        if self.config["fit.best_model_path"] is not None:
            # copy the best model_path
            if self.trainer.checkpoint_callback.best_model_path == "":
                logging.error(
                    "No best model path found. Please check if the checkpoint callback is enabled."
                )
            else:
                # Make sure file does not exist
                if self.config["fit.best_model_path"].exists():
                    logging.error(
                        f"File {self.config['fit.best_model_path']} already exists. Not copying {self.trainer.checkpoint_callback.best_model_path}."
                    )
                shutil.copy(
                    self.trainer.checkpoint_callback.best_model_path,
                    self.config["fit.best_model_path"],
                )


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
        LOG_DIR = PROJECT_DIR / "results" / "model_training"

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
                    save_dir=get_path(["paths", "wandb_logs"]),
                    project="JointEmbed_Training",
                    entity="single-cellm",
                    log_model=False,
                ),
            },
            enable_progress_bar=True,
            callbacks=[checkpoint_callback, early_stop],
        ),
        save_config_callback=None,
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    cli_main()
