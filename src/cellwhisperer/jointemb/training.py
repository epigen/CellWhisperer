#!/usr/bin/env python
# coding: utf-8
"""
See https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html for documentation on usage
"""
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.strategies import FSDPStrategy
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.biogpt.modeling_biogpt import BioGptDecoderLayer
from cellwhisperer.config import get_path, model_path_from_name, config
from cellwhisperer.misc.utils import obj_signature
from cellwhisperer.misc.debug import start_debugger
from scgpt.model import FlashTransformerEncoderLayer
import torch
from torch.nn import TransformerEncoderLayer
from typing import Optional, List
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
import subprocess
import shutil
import warnings
from pathlib import Path
import os
import yaml
import logging
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import Logger, WandbLogger
from jsonargparse import lazy_instance
import argparse

from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderConfig
from cellwhisperer.jointemb.loss.config import LossConfig
from cellwhisperer.jointemb.cellwhisperer_lightning import (
    TranscriptomeTextDualEncoderLightning,
)
from cellwhisperer.jointemb.dataset import JointEmbedDataModule

logger = logging.getLogger(__name__)


class CellWhispererCLI(LightningCLI):
    """CLI for cellwhisperer."""

    def __init__(self, *args, **kwargs):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings(
            "ignore", category=ResourceWarning
        )  # "ResourceWarning: Implicitly cleaning up <TemporaryDirectory..."

        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--model_ckpt",
            type=Path,
            default=None,
            help="Path to a checkpoint to load. This might override some of the other args (to be tested). Difference to --ckpt_path is that it does not restore the trainer/optimizer state.",
        )
        parser.add_argument("--best_model_path", type=Path, default=None)
        parser.add_argument("--last_model_path", type=Path, default=None)
        parser.add_argument("--log_level", default="INFO")
        parser.add_argument("--dap_debug", action="store_true")
        parser.add_argument(
            "--batch_size",
            default=32,
            type=lambda x: int(x)
            if int(x) > 1
            else argparse.ArgumentTypeError("Batch size must be greater than 1"),
            help="Batch size for training and evaluation.",
        )
        parser.add_argument(
            "--wandb",
            default="",
            help="Enable WandB logging. Need to pass a run name",
            type=str,
        )
        model_config_default = obj_signature(TranscriptomeTextDualEncoderConfig)
        parser.set_defaults(
            {
                "model.model_config": model_config_default,
                "model.loss_config": obj_signature(LossConfig),
                "trainer.max_epochs": 5,
                "trainer.reload_dataloaders_every_n_epochs": 1,  # this allows sampling of replicates (see data module)
                "trainer.accumulate_grad_batches": 64,  # the higher the better more or less
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

        parser.link_arguments(
            "model.model_config",
            "data.transcriptome_processor",
            lambda model_config: model_config.get(
                "transcriptome_model_type",
                model_config_default["transcriptome_model_type"],
            ),
        )

        parser.link_arguments(
            "model.model_config",
            "data.tokenizer",
            lambda model_config: model_config.get(
                "text_model_type",
                model_config_default["text_model_type"],
            ),
        )

        # for development: if fast_dev_run is enabled, set batch_size to 2
        batch_size_fn = (
            lambda fast_dev_run, batch_size: 2  # 1 fails
            if fast_dev_run or batch_size <= 0
            else batch_size
        )
        parser.link_arguments(
            ["trainer.fast_dev_run", "batch_size"],
            "data.batch_size",
            compute_fn=batch_size_fn,
        )

        # NOTE: crashed with large batch sizes
        # parser.link_arguments(
        #     ["trainer.fast_dev_run", "batch_size"],
        #     "model.val_batch_size",
        #     compute_fn=batch_size_fn,
        # )

    def before_instantiate_classes(self) -> None:
        if "fit.log_level" in self.config:
            log_level = self.config["fit.log_level"]
        elif "test.log_level" in self.config:
            log_level = self.config["test.log_level"]
        else:
            raise ValueError("No log level found")
        logging.basicConfig(level=log_level.upper())

        if "fit.dap_debug" in self.config and self.config["fit.dap_debug"]:
            start_debugger(wait_for_client=True)
        if "test.dap_debug" in self.config and self.config["test.dap_debug"]:
            start_debugger(wait_for_client=True)

    def before_fit(self) -> None:
        if not self.config["fit.ckpt_path"] and not self.config["fit.model_ckpt"]:
            # We need to preload this model
            if self.model.model.config.locking_mode[0] == "u":
                transcriptome_model_path = None
            else:
                transcriptome_model_path = model_path_from_name(
                    self.model.model.transcriptome_model.config.model_type
                )

            if self.model.model.config.locking_mode[1] == "u":
                text_model_path = None
            else:
                text_model_path = model_path_from_name(
                    self.model.model.text_model.config.model_type
                )

            if (
                self.model.model.config.locking_mode[0] != "L"
                and self.model.model.transcriptome_model.config.model_type == "scgpt"
            ):
                #  NOTE: because of FSDP being incapable of implicitly handling fp16 and fp32 conversion, we need to use scGPT without flash-attention and with 32 bit
                logger.warning(
                    "scgpt requireds 32 bit (and in consequence not flash attention). make sure that scgpt_config.fast_transformer is False"
                )

            try:
                self.model.load_pretrained_models(
                    transcriptome_model_path,
                    text_model_path,
                )
            except FileNotFoundError:
                logger.error(
                    "Unable to find the transcriptome model. Please download first (see `rna` snakemake pipeline). For scGPT: https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
                )
                raise

            if not bool(self.config["fit.batch_size"]):
                tuner = Tuner(self.trainer)
                tuner.scale_batch_size(
                    self.model, datamodule=self.datamodule, mode="binsearch", init_val=8
                )  # requires batch_size argument in datamodule or model
        elif self.config["fit.model_ckpt"]:  # model loading needs to be done implicitly
            logger.warning("Loading model from checkpoint. All other args are ignored")
            self.model = TranscriptomeTextDualEncoderLightning.load_from_checkpoint(
                self.config["fit.model_ckpt"]
            )

        # Optional: Log gradients
        # self.trainer.logger.watch(self.model, log="gradients")

    def before_test(self) -> None:
        if not self.config["test.model_ckpt"]:
            raise ValueError(
                "No checkpoint path found. Please provide a checkpoint path via --model_ckpt."
            )

        elif self.config[
            "test.model_ckpt"
        ]:  # model loading needs to be done implicitly
            logger.warning("Loading model from checkpoint. All other args are ignored")
            self.model = TranscriptomeTextDualEncoderLightning.load_from_checkpoint(
                self.config["test.model_ckpt"]
            )

    def after_fit(self) -> None:
        if self.config["fit.best_model_path"] is not None:
            # copy the best model_path
            if self.trainer.checkpoint_callback.best_model_path == "":
                logger.error(
                    "No best model path found. Please check if the checkpoint callback is enabled."
                )
            else:
                # Make sure file does not exist
                if self.config["fit.best_model_path"].exists():
                    logger.error(
                        f"File {self.config['fit.best_model_path']} already exists. Overwriting {self.trainer.checkpoint_callback.best_model_path}."
                    )
                shutil.copy(
                    self.trainer.checkpoint_callback.best_model_path,
                    self.config["fit.best_model_path"],
                )

        if self.config["fit.last_model_path"] is not None:
            # copy the last model_path
            if self.trainer.checkpoint_callback.last_model_path == "":
                logger.error(
                    "No last model path found. Please check if the checkpoint callback is enabled."
                )
            else:
                # Make sure file does not exist
                if self.config["fit.last_model_path"].exists():
                    logger.error(
                        f"File {self.config['fit.last_model_path']} already exists. Overwriting {self.trainer.checkpoint_callback.last_model_path}."
                    )
                shutil.copy(
                    self.trainer.checkpoint_callback.last_model_path,
                    self.config["fit.last_model_path"],
                )


class LoggerSaveConfigCallback(SaveConfigCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, overwrite=True, **kwargs)

    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(
                self.config, skip_none=False
            )  # Required for proper reproducibility
            trainer.logger.log_hyperparams(self.config.as_dict())


def cli_main(args: Optional[List] = None):
    """
    Args:
        args: Arguments to be used instead of sys.argv for LightningCLI. If None, sys.argv is used.
    """
    torch.set_float32_matmul_precision("high")  # speed up on ampere-level GPUs
    torch.multiprocessing.set_sharing_strategy("file_system")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable warning from tokenizers

    LOG_DIR = os.path.relpath(
        config["PROJECT_ROOT"] / "results" / "model_training", os.getcwd()
    )

    val_metric = "valfn_human_disease_strictly_deduplicated_dmis-lab_biobert-v1.1_CLS_pooling/text_as_classes_recall_at_5_macroAvg"

    # early_stop = EarlyStopping(
    #     monitor=val_metric, min_delta=1e-4, patience=10, verbose=False, mode="max"
    # )

    checkpoint_callback = ModelCheckpoint(
        monitor=val_metric,
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="{epoch}-valfn_human_disease_recall10={%s:.2f}" % (val_metric,),
    )

    CellWhispererCLI(
        TranscriptomeTextDualEncoderLightning,
        JointEmbedDataModule,
        trainer_defaults=dict(
            default_root_dir=LOG_DIR,
            precision="bf16-mixed",
            # NOTE: Activation checkpointing may reduce memory consumption. But it did not help much in the end
            # strategy={
            #     "class_path": "lightning.pytorch.strategies.FSDPStrategy",
            #     "init_args": {
            #         "activation_checkpointing_policy": {
            #             BioGptDecoderLayer,
            #             BertLayer,
            #             TransformerEncoderLayer,  # scGPT
            #             FlashTransformerEncoderLayer,  # scGPT
            #         },
            #         "sharding_strategy": "NO_SHARD",  # corresponds to DDP. no need to go fancy for the moment.. We can try later
            #     },
            # },
            logger={
                "class_path": WandbLogger.__module__ + "." + WandbLogger.__name__,
                "init_args": dict(
                    save_dir=os.path.relpath(
                        get_path(["paths", "wandb_logs"]), os.getcwd()
                    ),
                    project="JointEmbed_Training",
                    entity="single-cellm",
                    log_model=False,
                ),
            },
            enable_progress_bar=True,
            callbacks=[checkpoint_callback],
        ),
        save_config_callback=LoggerSaveConfigCallback,
        auto_configure_optimizers=False,
        args=args,
    )


if __name__ == "__main__":
    cli_main()
