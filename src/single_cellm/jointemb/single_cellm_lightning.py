"""
Wraps lightning and transformers logic, providing bare configs to the user/CLI
"""

from lightning import LightningModule
from single_cellm.jointemb.geneformer_model import GeneformerConfig
from .model import (
    TranscriptomeTextDualEncoderConfig,
    TranscriptomeTextDualEncoderModel,
    CLIPOutput,
)
from .loss import clip_loss

from typing import Optional, Union, Dict
from pathlib import Path
import torch
import copy

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
    ):
        """
        Args:
            dim: dimension of the projection layer
            logit_scale_init_value: initial value of the logit scale parameter (see CLIP paper)
        """

        model_config = copy.deepcopy(
            model_config
        )  # make sure not to modify the default arg itself

        super(TranscriptomeTextDualEncoderLightning, self).__init__()

        if not isinstance(model_config, TranscriptomeTextDualEncoderConfig):
            model_config = TranscriptomeTextDualEncoderConfig(**model_config)

        self.model = TranscriptomeTextDualEncoderModel(
            config=model_config,
        )

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
        return self.model.model(
            input_ids=input_ids,
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            attention_mask=attention_mask,
            return_dict=dict,
        )

    def process_step(self, batch, batch_idx, step_type):
        outputs = self.model(**batch, return_dict=True)

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

    def on_train_end(self):
        self.model.store_cache()
