"""
Wraps lightning and transformers logic, providing bare configs to the user/CLI
"""

import warnings

from lightning import LightningModule
from single_cellm.validation import TRAINING_VALIDATION_FUNCTIONS
from single_cellm.jointemb.loss.config import LossConfig
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from single_cellm.jointemb.model import (
    TranscriptomeTextDualEncoderConfig,
    TranscriptomeTextDualEncoderModel,
    CLIPOutput,
)

from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable


from typing import Optional, Union, Dict, List
from pathlib import Path
import torch
import copy
from functools import partial
import logging

from single_cellm.jointemb.regularization import InputRegularization

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*is deprecated and will be removed in a future version.*",
)


class TranscriptomeTextDualEncoderLightning(LightningModule):
    def __init__(
        self,
        model_config: Union[Dict, TranscriptomeTextDualEncoderConfig],
        loss_config: Union[Dict, LossConfig],
        gauss_noise_std: float = 0.0,
        max_epochs: int = 100,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        scheduler: LRSchedulerCallable = CosineAnnealingLR,
        learning_rate: float = 1e-3,
    ):
        """
        Args:
            model_config: configuration for the model. Can be a dict or a TranscriptomeTextDualEncoderConfig object.
            loss_config: configuration for the loss. Can be a dict or a LossConfig object.
            gauss_noise_std: Currently inactive: standard deviation of the gaussian noise to add to the input embeddings (features). if 0.0 or None noise won't be added to training embeddings
            max_epochs: maximum number of epochs to train for
            optimizer: optimizer to use. Must be a callable that returns an optimizer object.
            scheduler: scheduler to use. Must be a callable that returns a scheduler object.
            learning_rate: learning rate to use for the optimizer
        """
        super(TranscriptomeTextDualEncoderLightning, self).__init__()

        if not isinstance(model_config, TranscriptomeTextDualEncoderConfig):
            model_config = TranscriptomeTextDualEncoderConfig(**model_config)

        if not isinstance(loss_config, LossConfig):
            loss_config = LossConfig(**loss_config)

        self.model = TranscriptomeTextDualEncoderModel(
            config=model_config,
        )
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.learning_rate = learning_rate

        self.loss_config = loss_config
        self.loss_functions = self.loss_config.configure_losses(
            self.model.discriminator
        )

        self.input_regularization = InputRegularization(gauss_noise_std=gauss_noise_std)

        self.save_hyperparameters()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        geneformer_directory: str,
        text_model_name_or_path: str,
        **kwargs,
    ):
        """
        Because our checkpoints might not contain the foundation models, we need to load them separately and then load the checkpoint.

        TODO: we only need to do this if one of the models is frozen (which is usually the case)
        """
        model = super().load_from_checkpoint(checkpoint_path, **kwargs)
        # Park state_dict
        state_dict = model.state_dict().copy()

        # make sure that pretrained models are loaded
        model.load_pretrained_models(
            geneformer_directory=geneformer_directory,
            text_model_name_or_path=text_model_name_or_path,
        )

        # Restore state_dict
        model.load_state_dict(state_dict)

        return model

    def load_pretrained_models(
        self, geneformer_directory: str, text_model_name_or_path: str
    ):
        """
        This method exhibits an interface to load the pretrained models after initialization. This allows loading of pretrained weights from a checkpoint, without initializing these models..
        """
        self.model = (
            TranscriptomeTextDualEncoderModel.from_transcriptome_text_pretrained(
                transcriptome_model_name_or_path=geneformer_directory,
                text_model_name_or_path=text_model_name_or_path,
                **self.model.config.to_dict(),
            )
        )
        self.loss_functions = self.loss_config.configure_losses(
            self.model.discriminator
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        expression_tokens: Optional[torch.FloatTensor],
        expression_token_lengths: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
    ) -> CLIPOutput:
        # TODO at a later stage, we may add regularization here

        return self.model(
            input_ids=input_ids,
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            attention_mask=attention_mask,
            return_dict=True,
        )

    def process_step(self, batch, batch_idx, step_type):
        outputs = self(**batch)

        # outputs = {k: v.to(device) if v is not None else None for k, v in outputs.items()}
        combined_loss = torch.tensor(0.0, device=self.device)

        for loss in self.loss_functions:
            # Calculate the loss for the current batch using the specific loss function.
            loss_value = loss["fn"](**outputs)
            combined_loss = combined_loss + (loss_value * loss["lambda"])

            # Log the individual loss value for monitoring.
            self.log(
                f"{step_type}_{loss['name']}_loss",
                loss_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,  # NOTE: might lead to more synchronization overhead
            )

        # After processing all loss functions, log the total combined loss.
        self.log(
            f"{step_type}_loss",
            combined_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,  # NOTE: might lead to more synchronization overhead
        )
        # TODO: this is a hack to avoid NaNs in the loss. We should fix this properly.
        if torch.isnan(combined_loss):
            logging.warning("NaN loss detected. Setting loss to 0.0.")
            return torch.tensor(0.0, device=self.device)

        return combined_loss

    def training_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        # epoch_average = torch.stack(self.validation_step_outputs).mean()
        # self.log("validation_epoch_average", epoch_average)
        for val_fn_name, val_fn in TRAINING_VALIDATION_FUNCTIONS.items():
            val_metrics, results_df = val_fn(self.model)
            for metric_name, metric_value in val_metrics.items():
                self.log(
                    f"validation_{val_fn_name}_{metric_name}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

    def test_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        """
        Use AdamW optimizer for decoupled weight decay. Apply decay weights only (e.g. not biases), as indicated by the CLIP paper(s).

        Also implement cosine learning rate schedule, as indicated by the CLIP paper(s).
        """
        weight_params = [
            param for name, param in self.named_parameters() if "weight" in name
        ]
        other_params = [
            param for name, param in self.named_parameters() if "weight" not in name
        ]

        optimizer = self.optimizer(
            [
                {"params": weight_params, "weight_decay": 0.01},
                {"params": other_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
        )
        scheduler = {
            "scheduler": self.scheduler(optimizer, T_max=self.max_epochs, eta_min=0),
            "interval": "epoch",
            "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_end(self):
        self.model.store_cache()
