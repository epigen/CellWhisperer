"""
Wraps lightning and transformers logic, providing bare configs to the user/CLI
"""

import warnings

from lightning import LightningModule
from single_cellm.validation import TRAINING_VALIDATION_FUNCTIONS
from single_cellm.jointemb.loss.config import LossConfig
from single_cellm.jointemb.model import (
    TranscriptomeTextDualEncoderConfig,
    TranscriptomeTextDualEncoderModel,
    CLIPOutput,
)

# from single_cellm.jointemb.config import LossConfig ## NOT WORKING

from typing import Optional, Union, Dict, List
from pathlib import Path
import torch
import copy
from functools import partial

from single_cellm.jointemb.regularization import InputRegularization

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*is deprecated and will be removed in a future version.*",
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL_CONFIG = TranscriptomeTextDualEncoderConfig(
    transcriptome_config={"model_type": "geneformer"},
    text_config={"model_type": "biogpt"},
).to_dict()


class TranscriptomeTextDualEncoderLightning(LightningModule):
    def __init__(
        self,
        model_config: Union[Dict, TranscriptomeTextDualEncoderConfig] = copy.deepcopy(
            DEFAULT_MODEL_CONFIG
        ),  # needs copy to avoid GPU memory leak
        loss_config: Union[Dict, LossConfig] = LossConfig().to_dict(),
        gauss_noise_std: float = 0.0,  # Noise standard deviation: if 0.0 or None noise won't be added to training embeddings
    ):
        """
        Args:
            dim: dimension of the projection layer
            logit_scale_init_value: initial value of the logit scale parameter (see CLIP paper)
        """

        model_config = copy.deepcopy(
            model_config
        )  # make sure not to modify the default arg itself

        loss_config = copy.deepcopy(loss_config)

        super(TranscriptomeTextDualEncoderLightning, self).__init__()

        if not isinstance(model_config, TranscriptomeTextDualEncoderConfig):
            model_config = TranscriptomeTextDualEncoderConfig(**model_config)

        if not isinstance(loss_config, LossConfig):
            loss_config = LossConfig(**loss_config)

        self.model = TranscriptomeTextDualEncoderModel(
            config=model_config,
        )

        self.loss_config = loss_config
        self.loss_functions = self.loss_config.configure_losses(
            self.model.projection_dim, self.model.discriminator
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
            self.model.projection_dim, self.model.discriminator
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
        combined_loss = 0.0

        for loss_name, loss_fn in self.loss_functions.items():
            # Calculate the loss for the current batch using the specific loss function.
            loss_value = loss_fn(**outputs)
            combined_loss += loss_value

            # Log the individual loss value for monitoring.
            self.log(
                f"{step_type}_{loss_name}_loss",
                loss_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        # After processing all loss functions, log the total combined loss.
        self.log(
            f"{step_type}_loss",
            combined_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        return optimizer

    def on_train_end(self):
        self.model.store_cache()

    def on_validation_end(self):
        self.model.store_cache()
