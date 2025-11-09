import numpy as np
import pandas as pd
import torch
from cellwhisperer.utils.inference import score_transcriptomes_vs_texts
from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel
from cellwhisperer.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from cellwhisperer.jointemb.scgpt_model import ScGPTTranscriptomeProcessor
from cellwhisperer.jointemb.uce_model import UCETranscriptomeProcessor
from transformers import AutoTokenizer
import anndata
from typing import Tuple, Union, List, Dict, Optional
import torchmetrics
from copy import copy
from typing import Dict
import numpy as np


def get_performance_metrics_transcriptome_vs_text(
    model: TranscriptomeTextDualEncoderModel,
    transcriptome_input: Union[anndata.AnnData, torch.Tensor],
    text_list_or_text_embeds: Union[List[str], torch.Tensor],
    correct_text_idx_per_transcriptome: List[int],
    average_mode: Optional[str] = "embeddings",
    grouping_keys: Optional[List[str]] = None,
    transcriptome_processor: Optional[
        Union[GeneformerTranscriptomeProcessor, ScGPTTranscriptomeProcessor, UCETranscriptomeProcessor]
    ] = None,
    batch_size: int = 128,
    score_norm_method: Optional[str] = None,
    report_per_class_metrics: bool = True,
    text_as_classes: bool = True,
) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame]:
    """
    Score the model's ability to produce similar embeddings for the given matching texts and adata objects.
    :param model: TranscriptomeTextDualEncoderModel instance. Both transcriptome and text embeddings will be computed using this model.
    :param transcriptome_input: anndata.AnnData or torch.tensor (n_cells*embedding_size) \
        If anndata.AnnData, first compute the transcriptome embeddings. If torch.tensor, use the provided transcriptome embeddings.
    :param text_list_or_text_embeds: List[str] or torch.tensor (n_celltype * embedding_size). If List[str], compute the text embeddings for each text. \
        If torch.tensor, use the provided text embeddings.
    :param correct_text_idx_per_transcriptome: A list with the index in text_list_or_text_embeds of the correct text for each transcriptome in transcriptome_input.
    :param average_mode: "embeddings" or None. \
        If "embeddings", first tokenize and embed each cell, then average the embeddings. If None, don't average, report results at the single-transcriptome level. NOTE "cells" is not implemented at the moment but would work by first averaging the transcriptome data across all cells of same celltype, then tokenize and embed. \
    :param grouping_keys: A list with group indicators (one for each transcriptome in transcriptome_input). If average_mode is not None, this must be provided and will be used to split the transciptome_input into groups that will be averaged separately.
          Will also be used to label the rows in the result dataframe. If None, just use numbers.
    :param transcriptome_processor: GeneformerTranscriptomeProcessor, UCETranscriptomeProcessor or ScGPTTranscriptomeProcessor instance. Used to prepare/tokenize the transcriptome. Can be None if transcriptome_input is a torch.tensor.
    :param batch_size: int. Model processing in batches (to avoid OOM)
    :param score_norm_method: "zscore", "softmax", "01norm" or None. How to normalize the logits \
            (similarity to the transcriptome). "zscore" will zscore the logits across all terms. "softmax" will apply softmax to the logits.\
            "01norm" will normalize the logits to the range [0,1]. If None, don't normalize.
    :param report_per_class_metrics: bool. If True, report the performance metrics per class (i.e. per transcriptome). If False, only report the macro average.
    : param text_as_classes: bool. If True, calculate the score using the text as classes (default). \
        If False, calculate the score using the transcriptome as classes (can be useful for retrieval scoring). \
        If False, average_mode must be None.
    Returns: 
        A tuple of: 
         -  A dictionary containing precision, recall (at k=1,5,10,50), accuracy, f1, and rocauc. \
            These metrics are reported using macro averaging (i.e. averaging over classes)
         -  None if report_per_class_metrics=False, else a dataframe with cell type as rows and performance metrics as columns. \
    """

    full_grouping_keys = copy(
        grouping_keys
    )  # We keep the full annotations in case they get modified due to averaging below

    # Get the scores.
    # grouping_keys will be updated (deduplicated and put into correct order) if averaging is used, else kept the same.
    scores, grouping_keys = score_transcriptomes_vs_texts(
        transcriptome_input=transcriptome_input,
        text_list_or_text_embeds=text_list_or_text_embeds,
        model=model,
        logit_scale=model.discriminator.temperature.exp(),
        average_mode=average_mode,
        grouping_keys=grouping_keys,
        transcriptome_processor=transcriptome_processor,
        batch_size=batch_size,
        score_norm_method=score_norm_method,
    )  # scores is a tensor of shape n_text * n_cells (if average_mode is None), or n_text * n_celltypes (otherwise).
    # It is normalized column-wise, i.e. for each adata, the mean of scores is 0 (if using zscore normalization)

    # labels for the transcriptome, and the true classes
    if grouping_keys is None:
        grouping_keys = [str(x) for x in range(scores.shape[1])]
        if average_mode is not None:
            raise ValueError(
                "If grouping_keys is None, average_mode must be None as well"
            )
        else:
            true_class_indices = correct_text_idx_per_transcriptome
    elif not text_as_classes:
        # the correct transcriptome index for every text:
        true_class_indices = [
            correct_text_idx_per_transcriptome.index(x)
            for x in range(len(correct_text_idx_per_transcriptome))
        ]
    else:
        if average_mode is not None:
            true_class_indices = [
                correct_text_idx_per_transcriptome[full_grouping_keys.index(x)]
                for x in grouping_keys
            ]  # If we are averaging, we need to subset the true classes to match the averaged transcriptomes
        else:
            # we can just use the annotations and true classes as they are
            true_class_indices = correct_text_idx_per_transcriptome

    # labels for the text
    if type(text_list_or_text_embeds) == torch.Tensor:
        text_annotations = [str(x) for x in range(scores.shape[0])]
    else:
        text_annotations = text_list_or_text_embeds

    text_labels = [f"text: {x}" for x in text_annotations]
    transcriptome_labels = [f"transcriptome: {x}" for x in grouping_keys]

    if text_as_classes:
        scores = (
            scores.t()
        )  # Here, it's better to have samples=transcriptomes, classes=texts, so we transpose
        columns = text_labels
        index = transcriptome_labels
        num_classes = int(max(true_class_indices) + 1)

    else:
        if average_mode is not None:
            raise ValueError(
                "If text_as_classes is False, average_mode must be None as well"
            )
        columns = transcriptome_labels
        index = text_labels
        num_classes = len(grouping_keys)

    # Create a dataframe with the scores
    scores_df = pd.DataFrame(
        scores.cpu().numpy(),
        columns=columns,
        index=index,
    )
    if average_mode is not None and type(text_list_or_text_embeds) != torch.Tensor:
        scores_df = scores_df[sorted(scores_df.columns)]

    torchmetric_kwargs = {
        "preds": scores,
        "target": torch.Tensor(true_class_indices).long(),
        "num_classes": num_classes,
        "average": "none",
        "top_k": 1,
    }

    precision = torchmetrics.functional.classification.multiclass_precision(
        **torchmetric_kwargs
    )
    accuracy = torchmetrics.functional.classification.multiclass_accuracy(
        **torchmetric_kwargs
    )
    f1 = torchmetrics.functional.classification.multiclass_f1_score(
        **torchmetric_kwargs
    )
    rocauc = torchmetrics.functional.classification.multiclass_auroc(
        **{k: v for k, v in torchmetric_kwargs.items() if not k == "top_k"}
    )

    confusion_mtx = torchmetrics.functional.classification.multiclass_confusion_matrix(
        **{k: v for k, v in torchmetric_kwargs.items() if k not in ["top_k", "average"]}
    )

    res_metrics = {
        "precision": precision.detach(),
        "accuracy": accuracy.detach(),
        "f1": f1.detach(),
        "rocauc": rocauc.detach(),
    }

    for k in [1, 5, 10, 50]:
        if num_classes >= k:
            torchmetric_kwargs.update({"top_k": k})
            res_metrics[
                f"recall_at_{k}"
            ] = torchmetrics.functional.classification.multiclass_recall(
                **torchmetric_kwargs
            ).detach()
            torchmetric_kwargs.update({"top_k": 1})
        else:
            res_metrics[f"recall_at_{k}"] = torch.tensor([np.nan] * num_classes)

    macro_average_results = {}
    for metric, value in res_metrics.items():
        macro_average_results[f"{metric}_macroAvg"] = value.mean()

    # Create a dataframe with per-class metrics. Rows: classes, columns: metrics. Rownames: class names, column names: metric names
    if report_per_class_metrics:
        per_class_df = pd.DataFrame(
            torch.stack([value for value in res_metrics.values()], dim=1).numpy(),
            columns=[x for x in res_metrics.keys()],
            index=text_annotations,
        )
        per_class_df.index.name = "class"
        per_class_df["n_samples_in_class"] = [
            true_class_indices.count(class_idx)
            for class_idx, _ in enumerate(text_annotations)
        ]

        confusion_df = pd.DataFrame(
            confusion_mtx.numpy(),
            index=text_annotations,
            columns=[f"n_samples_predicted_as_{x}" for x in text_annotations],
        )
        confusion_df.index.name = "class"
        per_class_df = pd.concat([per_class_df, confusion_df], axis=1)

    else:
        per_class_df = None

    return (
        macro_average_results,
        per_class_df,
    )
