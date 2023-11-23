"""
Wraps lightning and transformers logic, providing bare configs to the user/CLI
"""

from lightning import LightningModule
from single_cellm.jointemb.geneformer_model import GeneformerConfig
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
        loss_configs: Union[Dict, LossConfig] = LossConfig().to_dict(),
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

        loss_configs = copy.deepcopy(loss_configs)

        super(TranscriptomeTextDualEncoderLightning, self).__init__()

        if not isinstance(model_config, TranscriptomeTextDualEncoderConfig):
            model_config = TranscriptomeTextDualEncoderConfig(**model_config)

        if not isinstance(loss_configs, LossConfig):
            loss_configs = LossConfig(**loss_configs)

        self.model = TranscriptomeTextDualEncoderModel(
            config=model_config,
        )

        self.loss_functions = loss_configs.configure_losses(model_config.projection_dim)

        self.input_regularization = InputRegularization(gauss_noise_std=gauss_noise_std)

        self.save_hyperparameters()

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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        expression_tokens: Optional[torch.FloatTensor],
        expression_token_lengths: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
    ) -> CLIPOutput:
        # TODO at a later stage, we may add regularization here

        return self.model.model(
            input_ids=input_ids,
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            attention_mask=attention_mask,
            return_dict=True,
        )

    def process_step(self, batch, batch_idx, step_type):
        outputs = self.model(**batch, return_dict=True)
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

    def test_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        return optimizer

    def on_train_end(self):
        self.model.store_cache()

    def on_validation_end(self):
        self.model.store_cache()
