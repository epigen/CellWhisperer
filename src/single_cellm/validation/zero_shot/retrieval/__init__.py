from single_cellm.validation.zero_shot.functions import (
    get_performance_metrics_transcriptome_vs_text,
)
from transformers import AutoTokenizer
import anndata
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import torch


class RetrievalScoreCalculator:
    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader = None,
        max_n_samples=1000,
    ):
        """
        Class to calculate retrieval scores for a given dataset with transcriptome+text information. Each transcriptome is scored against each text, \
            and we check if the correct text is in the top 1,5,or 10 best matches.
        Args:
            max_n_samples: Maximum number of samples to process.
            dataloader: if not None, the dataloader to use. If None, a new one is created based on the other arguments.
        """

        self.dataloader = dataloader
        self.batch_size = self.dataloader.batch_size
        self.max_n_samples = max_n_samples

    def __call__(self, model) -> Tuple[Dict[str, float], pd.DataFrame]:
        text_embeds = []
        transcriptome_embeds = []

        for i, batch in enumerate(self.dataloader):
            if i * self.batch_size >= self.max_n_samples:
                break
            batch = {k: v.to(model.device) for k, v in batch.items()}
            res = model(**batch)
            res.text_embeds = res.text_embeds.detach()  # .cpu()
            res.transcriptome_embeds = res.transcriptome_embeds.detach()  # .cpu()
            text_embeds.append(res["text_embeds"])
            transcriptome_embeds.append(res["transcriptome_embeds"])

        text_embeds = torch.cat(text_embeds)
        transcriptome_embeds = torch.cat(transcriptome_embeds)

        result_dict, result_df = get_performance_metrics_transcriptome_vs_text(
            transcriptome_input=transcriptome_embeds,
            model=model,
            text_tokenizer=None,
            transcriptome_processor=None,
            average_mode=None,  # We treat each transcriptome separately
            text_list_or_text_embeds=text_embeds,
            batch_size=self.batch_size,
        )

        return result_dict, result_df
