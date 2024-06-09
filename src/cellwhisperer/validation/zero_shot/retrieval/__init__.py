import torch
from cellwhisperer.validation.zero_shot.functions import (
    get_performance_metrics_transcriptome_vs_text,
)
import pandas as pd
from typing import Dict, Tuple, Union


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
        """
        Args:
            model: a trained model
        Returns:
            A tuple of: 
            - A dictionary containing macro-averaged precision, recall (at k=1,5,10,50), accuracy, f1, and rocauc. These metrics are reported \
                seperately for using text as classes (and therefore transcriptomes as samples) and for using transcriptomes as classes (and text as samples).
            - None (reporting per-class metrics doesn't make much sense here without a consistent short label for each transcriptome)
        """
        text_embeds = []
        transcriptome_embeds = []

        for i, batch in enumerate(self.dataloader):
            if i * self.batch_size >= self.max_n_samples:
                break
            batch = {
                k: v.to(model.device)
                for k, v in batch.items()
                if k not in ["transcriptome_weights", "annotation_weights"]
            }
            res = model(**batch)
            res.text_embeds = res.text_embeds.detach()  # .cpu()
            res.transcriptome_embeds = res.transcriptome_embeds.detach()  # .cpu()
            text_embeds.append(res["text_embeds"])
            transcriptome_embeds.append(res["transcriptome_embeds"])

        text_embeds = torch.cat(text_embeds)
        transcriptome_embeds = torch.cat(transcriptome_embeds)

        performance_metrics_all = {}
        for name, text_as_classes in zip(
            ["transcriptomes_as_classes", "text_as_classes"], [False, True]
        ):
            (
                performance_metrics,
                _,
            ) = get_performance_metrics_transcriptome_vs_text(
                transcriptome_input=transcriptome_embeds,
                model=model,
                transcriptome_processor=None,
                correct_text_idx_per_transcriptome=list(
                    range(text_embeds.shape[0])
                ),  # the text embeds are in the same order as the transcriptome embeds
                average_mode=None,  # We treat each transcriptome separately
                text_list_or_text_embeds=text_embeds,
                batch_size=self.batch_size,
                grouping_keys=None,
                report_per_class_metrics=False,  # doesn't make much sense without a consistent short label for each transcriptome
                text_as_classes=text_as_classes,
            )

            performance_metrics = {
                f"{name}_{k}": v for k, v in performance_metrics.items()
            }
            performance_metrics_all.update(performance_metrics)

        return (
            performance_metrics_all,
            None,
        )
