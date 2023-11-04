"""
Wraps lightning and transformers logic, providing bare configs to the user/CLI
"""

import lightning as pl
from .model import (
    TranscriptomeTextDualEncoderConfig,
    TranscriptomeTextDualEncoderModel,
    CLIPOutput,
)
from .loss import clip_loss

from typing import Optional, Union
from pathlib import Path
import torch


class TranscriptomeTextDualEncoderLightning(pl.LightningModule):
    def __init__(
        self,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        transcriptome_model_type="geneformer",
        text_model_type="biogpt",
    ):
        """
        Args:
            dim: dimension of the projection layer
            logit_scale_init_value: initial value of the logit scale parameter (see CLIP paper)
        """
        super(TranscriptomeTextDualEncoderLightning, self).__init__()

        config_transcriptome = {"model_type": transcriptome_model_type}
        config_text = {"model_type": text_model_type}

        model_config = TranscriptomeTextDualEncoderConfig(
            projection_dim=projection_dim,
            logit_scale_init_value=logit_scale_init_value,
            transcriptome_config=config_transcriptome,
            text_config=config_text,
        )

        self.model = TranscriptomeTextDualEncoderModel(
            config=model_config,
        )

        self.save_hyperparameters()

    def load_pretrained_models(self, geneformer_directory: Union[Path, str]):
        self.model.text_model = self.model.text_model.from_pretrained(
            "microsoft/biogpt"
        )

        self.model.transcriptome_model.geneformer_model = (
            self.model.transcriptome_model.geneformer_model.from_pretrained(
                str(geneformer_directory),
                output_hidden_states=True,
                output_attentions=False,
            )
        )

    def freeze_pretrained(self):
        # Freeze the pretrained models
        for param in self.model.text_model.parameters():
            param.requires_grad = False
        for param in self.model.transcriptome_model.parameters():
            param.requires_grad = False

    def on_save_checkpoint(self, checkpoint):
        """
        Drop frozen parameters (don't save them)
        """

        for param_name in [
            param_name
            for param_name, param in self.named_parameters()
            if not param.requires_grad
        ]:
            try:
                del checkpoint["state_dict"][f"{param_name}"]
            except KeyError:
                print(f"Key {param_name} not found")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        expression_tokens: Optional[torch.FloatTensor],
        expression_token_lengths: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
    ) -> CLIPOutput:
        return self.model.model(
            input_ids=input_ids,
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            attention_mask=attention_mask,
            return_dict=True,
        )

    def process_step(self, batch, batch_idx, step_type):
        outputs = self.model(**batch)
        loss = clip_loss(outputs.logits_per_text)
        self.log(
            f"{step_type}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        return optimizer
