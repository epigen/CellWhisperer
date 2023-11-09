import torch
from torch import nn
from torch.nn import functional as F
from single_cellm.config import model_path_from_name
from transformers import AutoTokenizer
from lightning import LightningModule


class LitCancerGeneEssentiality(LightningModule):
    def __init__(self, model, anchor_sentence: str = "cancer"):
        super().__init__()

        self.model = model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path_from_name(self.model.config.text_config.model_type)
        )
        # alterantive: tokenizer = self.trainer.datamodule.processor.tokenizer
        token_dict = tokenizer(anchor_sentence, return_tensors="pt", padding=True)
        with torch.no_grad():
            self.anchor_embed = self.model.get_text_features(
                **token_dict, normalize_embeds=True
            )[1]

    def predict_step(self, batch, batch_idx):
        """
        Later use forward() and test_step to directly compute the metric
        """
        labels = batch.pop("labels")

        with torch.no_grad():
            transcriptome_embeds = self.model.get_transcriptome_features(
                **batch, normalize_embeds=True
            )[1]

        similarity = torch.matmul(
            self.anchor_embed, transcriptome_embeds.t().to(self.anchor_embed.device)
        )  # TODO .to(device) is workaround that should not be necessary with lightning

        return {"labels": labels, "similarity": similarity.cpu()}
