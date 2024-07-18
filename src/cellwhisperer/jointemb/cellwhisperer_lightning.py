"""
Wraps lightning and transformers logic, providing bare configs to the user/CLI
"""

import warnings

from lightning import LightningModule
from cellwhisperer.validation import initialize_validation_functions
from cellwhisperer.config import model_path_from_name
from cellwhisperer.jointemb.loss.config import LossConfig
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from cellwhisperer.jointemb.model import (
    TranscriptomeTextDualEncoderConfig,
    TranscriptomeTextDualEncoderModel,
    CLIPOutput,
)

from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from wandb import Artifact, Table

from typing import Optional, Union, Dict, List
from pathlib import Path
import torch
import copy
from functools import partial
import logging

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*is deprecated and will be removed in a future version.*",
)


class TranscriptomeTextDualEncoderLightning(LightningModule):
    def __init__(
        self,
        model_config: Dict,
        loss_config: Dict,
        val_batch_size: int = 32,
        gauss_noise_std: float = 0.0,
        max_epochs: int = 16,
        learning_rate: float = 1e-3,
        lr_warmup: Union[int, float] = 0.03,
        frozen_warmup: Union[int, float] = 0.03,
    ):
        """
        Args:
            model_config: configuration for the model. Can be a dict or a TranscriptomeTextDualEncoderConfig object.
            loss_config: configuration for the loss. Can be a dict or a LossConfig object.
            val_batch_size: batch size to used for dedicated validation functions. 32 should be feasible on most GPUs.
            gauss_noise_std: Deprecated & unused. Will be removed in future versions.
            max_epochs: maximum number of epochs to train for
            learning_rate: learning rate to use for the optimizer.
            lr_warmup: number of steps to use for learning rate warmup. If set to 0, no warmup is used. If set to a float, it is interpreted as a fraction of the total number of steps.
            frozen_warmup: number of steps to use for pure projection layer training. If set to 0, no warmup is used. If set to a float, it is interpreted as a fraction of the total number of steps.
        """
        super(TranscriptomeTextDualEncoderLightning, self).__init__()

        # Convert dicts to the respective object types (checkpoints contain the Config object, but training passes dicts)
        if not isinstance(model_config, TranscriptomeTextDualEncoderConfig):
            model_config = TranscriptomeTextDualEncoderConfig(**model_config)

        if not isinstance(loss_config, LossConfig):
            loss_config = LossConfig(**loss_config)

        self.model = TranscriptomeTextDualEncoderModel(
            config=model_config,
        )
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.lr_warmup = lr_warmup
        self.warmup_reset_step = 0

        self.frozen_warmup = frozen_warmup

        self.loss_config = loss_config
        self.loss_functions = self.loss_config.configure_losses(
            self.model.discriminator
        )

        self.val_batch_size = val_batch_size

        self.save_hyperparameters()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        **kwargs,
    ):
        """
        Because our checkpoints might not contain the foundation models, we need to load them separately and then load the checkpoint.

        NOTE: we only need to do this explicit sub-model loading, if one of the models is frozen (which is usually the case)
        """
        model = super().load_from_checkpoint(checkpoint_path, **kwargs)
        # Park state_dict
        state_dict = model.state_dict().copy()

        # if transcriptome_model_directory is None:
        transcriptome_model_directory = model_path_from_name(
            model.model.transcriptome_model.config.model_type
        )
        # if text_model_name_or_path is None:
        text_model_name_or_path = model_path_from_name(
            model.model.text_model.config.model_type
        )

        # make sure that pretrained models are loaded
        model.load_pretrained_models(
            transcriptome_model_directory=transcriptome_model_directory,
            text_model_name_or_path=text_model_name_or_path,
        )

        # Restore state_dict
        model.load_state_dict(state_dict)

        return model

    def load_pretrained_models(
        self,
        transcriptome_model_directory: Optional[str],
        text_model_name_or_path: Optional[str],
    ):
        """
        This method exhibits an interface to load the pretrained models after initialization. This allows loading of pretrained weights from a checkpoint, without initializing these models..
        """

        kwargs = self.model.config.to_dict()
        if transcriptome_model_directory is None:
            kwargs["transcriptome_model"] = self.model.transcriptome_model

        if text_model_name_or_path is None:
            kwargs["text_model"] = self.model.text_model

        self.model = (
            TranscriptomeTextDualEncoderModel.from_transcriptome_text_pretrained(
                transcriptome_model_name_or_path=transcriptome_model_directory,
                text_model_name_or_path=text_model_name_or_path,
                **kwargs,
            )
        )
        self.loss_functions = self.loss_config.configure_losses(
            self.model.discriminator
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        expression_tokens: Optional[torch.FloatTensor] = None,
        expression_token_lengths: Optional[torch.LongTensor] = None,
        expression_gene: Optional[torch.LongTensor] = None,
        expression_expr: Optional[torch.LongTensor] = None,
        expression_key_padding_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        transcriptome_weights: Optional[torch.FloatTensor] = None,
        annotation_weights: Optional[torch.FloatTensor] = None,
        **kwargs,  # token_type_ids
    ) -> CLIPOutput:
        output = self.model(
            input_ids=input_ids,
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            expression_gene=expression_gene,
            expression_expr=expression_expr,
            expression_key_padding_mask=expression_key_padding_mask,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        output["transcriptome_weights"] = transcriptome_weights
        output["annotation_weights"] = annotation_weights
        return output

    def process_step(self, batch, batch_idx, step_type):
        outputs = self(**batch)

        if not self.loss_config.sample_weighting:
            outputs["transcriptome_weights"] = None
            outputs["annotation_weights"] = None

        # outputs = {k: v.to(device) if v is not None else None for k, v in outputs.items()}
        combined_loss = torch.tensor(0.0, device=self.device)

        for loss in self.loss_functions:
            # Calculate the loss for the current batch using the specific loss function.

            loss_value = loss["fn"](**outputs)
            combined_loss = combined_loss + (loss_value * loss["lambda"])

            # Log the individual loss value for monitoring.
            self.log(
                f"{step_type}/{loss['name']}_loss",
                loss_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        # After processing all loss functions, log the total combined loss.
        self.log(
            f"{step_type}/loss",
            combined_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        # NOTE: this is a hack to avoid NaNs in the loss. We should fix this properly.
        if torch.isnan(combined_loss):
            logger.warning("NaN loss detected. Setting loss to 0.0.")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return combined_loss

    def training_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "val")

    def setup(self, stage=None):
        # We do this here because during __init__ self.trainer is not yet available
        self.validation_functions = initialize_validation_functions(
            batch_size=self.val_batch_size,
            transcriptome_model_type=self.model.transcriptome_model.config.model_type,
            text_model_type=self.model.text_model.config.model_type,
        )
        if stage == "fit":
            if isinstance(self.frozen_warmup, float):
                self.frozen_warmup_steps = int(
                    len(self.trainer.datamodule.train_dataloader())
                    * self.max_epochs
                    * self.frozen_warmup
                    / self.trainer.accumulate_grad_batches
                )
            else:
                self.frozen_warmup_steps = self.frozen_warmup

    def on_fit_start(self):
        # freeze for first epoch to train only the projection layer
        self.model.freeze_models()

    def on_validation_epoch_end(self):
        # For convenience (speed), I disable this when "fast_dev_run" is enabled
        if not self.trainer.fast_dev_run:
            logger.info("Running validation functions")
            for val_fn_name, val_fn in self.validation_functions.items():
                with torch.no_grad():  # necessary, despite model being in eval mode
                    val_results = val_fn(self.model)
                    val_metrics = val_results[0]
                for metric_name, metric_value in val_metrics.items():
                    # NOTE enabling sync_dist requires the logged metric to be on GPU
                    # In our case it doesn't matter, because we are training on a single GPU/node at the moment
                    # see https://github.com/Lightning-AI/pytorch-lightning/issues/18803
                    self.log(
                        f"valfn_{val_fn_name}/{metric_name}",
                        metric_value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                        # sync_dist=True,  # needs tensor on gpu
                    )
                    if val_results[1] is not None:
                        artifact = Artifact(
                            f"valfn_{val_fn_name}_per_celltype_run{self.logger.experiment.id}",
                            type="performance_metrics",
                        )
                        # turn the df val_results[1] into a wandb table:
                        table = Table(dataframe=val_results[1].reset_index())
                        artifact.add(table, "performance_metrics")
                        self.logger.experiment.log_artifact(artifact)

    def on_train_batch_start(self, *args):
        if self.trainer.global_step >= self.frozen_warmup_steps:
            logger.debug("Unfreezing U towers")
            self.model.unfreeze_U_towers()

            logger.debug("Warmup again")
            self.warmup_reset_step = self.trainer.global_step

            # Reset the optimizer (doesn't work because of the scheduler. would need to find the correct function)
            # new_optimizers = self.configure_optimizers()
            # self.trainer.optimizers = [new_optimizers["optimizer"]]
            # self.trainer.lr_schedulers_configs = [new_optimizers["lr_scheduler"]]

            self.frozen_warmup_steps = 1e9  # never evaluate this code again

    def on_train_epoch_end(self):
        """
        This function is called at the end of each epoch, after training AND validation.

        Store the cache of the model, after the first epoch to make it available to other runs as early as possible.
        """

        if self.trainer.current_epoch == 0:
            logger.debug("Storing cache of model")
            self.model.store_cache()

    def on_train_end(self):
        """
        Store the cache of the model, at the end of training. Since we drop the last batch (in training), we don't get all samples in the first batch
        """
        logger.debug("Storing cache of model")
        self.model.store_cache()

    def test_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "test")

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        """
        Use AdamW optimizer for decoupled weight decay. Apply decay weights only (e.g. not biases), as indicated by the CLIP paper(s).
        In reality we train with L and U only, so no decay is applied anyways

        Also implement cosine learning rate schedule, as indicated by the CLIP paper(s).

        https://arxiv.org/pdf/2303.15343.pdf (Fig. 4) indicates better performance when NOT decaying the weights of the 'U'nlocked pretrained model. Other CLIP papers indicated that only the weigths not the bias should be decayed.

        Alternative could be to use AdaBelief (which works well according to "ItalianClip")
        """
        # Decay weights as indicated in docstring
        # NOTE subject this to ablation study
        decay_selector = lambda name: "weight" in name and (
            (
                self.model.config.locking_mode[0] == "u"
                and ".transcriptome_model." in name
            )
            or (self.model.config.locking_mode[1] == "u" and ".text_model." in name)
        )

        weight_params = [
            param for name, param in self.named_parameters() if decay_selector(name)
        ]
        other_params = [
            param for name, param in self.named_parameters() if not decay_selector(name)
        ]

        optimizer = AdamW(
            [
                {"params": weight_params, "weight_decay": 0.01},
                {"params": other_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
            betas=(0.9, 0.98),  # SigLiT recommends even 0.95 for beta2
        )

        if isinstance(self.lr_warmup, float):
            self.lr_warmup_steps = int(
                len(self.trainer.datamodule.train_dataloader())
                * self.max_epochs
                * self.lr_warmup
                / self.trainer.accumulate_grad_batches
            )
        else:
            self.lr_warmup_steps = self.lr_warmup
        logger.info("Using lr_warmup_steps: %d", self.lr_warmup_steps)

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=len(self.trainer.datamodule.train_dataloader())
            * self.max_epochs
            / self.trainer.accumulate_grad_batches,
            eta_min=self.learning_rate / 20,
        )

        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Override `optimizer_step` to allow learning-rate warmup
        """

        # If we are still in the warmup phase, overwrite learning rate
        current_step = self.trainer.global_step - self.warmup_reset_step

        if current_step < self.lr_warmup_steps:
            lr_scale = min(1.0, float(current_step + 1) / self.lr_warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.learning_rate

        # Call the original optimizer_step. (It's fine if it applies cosine scheduling there)
        super(TranscriptomeTextDualEncoderLightning, self).optimizer_step(
            epoch, batch_idx, optimizer, optimizer_closure
        )

        for pg in optimizer.param_groups:
            self.log(
                "train/learning_rate",
                pg["lr"],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            break  # (only the first one is needed)
