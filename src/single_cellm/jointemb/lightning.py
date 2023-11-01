import pytorch_lightning as pl
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
        config: TranscriptomeTextDualEncoderConfig,
        *args,
        **kwargs,
    ):
        super(TranscriptomeTextDualEncoderLightning, self).__init__()

        self.model = TranscriptomeTextDualEncoderModel(
            config=config,
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

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = clip_loss(outputs.logits_per_text)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # print(batch)
        outputs = self.model(**batch)
        loss = clip_loss(outputs.logits_per_text)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = clip_loss(outputs.logits_per_text)
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        return optimizer


# Example of how to train the model
# dataset = ... # Your Dataset
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# model = TranscriptomeTextDualEncoderLightning(...)
# trainer = pl.Trainer(max_epochs=5, gpus=1)
# trainer.fit(model, train_loader)
