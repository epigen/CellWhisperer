import numpy as np
import pandas as pd
import torch
from cellwhisperer.utils.inference import score_left_vs_right
from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel
from transformers import AutoTokenizer
import anndata
from typing import Tuple, Union, List, Dict, Optional
import torchmetrics
from copy import copy
from typing import Dict
import numpy as np


def get_performance_metrics_left_vs_right(
    model: TranscriptomeTextDualEncoderModel,
    left_input: Union[anndata.AnnData, torch.Tensor, List[str]],
    right_input: Union[anndata.AnnData, torch.Tensor, List[str]],
    correct_right_idx_per_left: List[int],
    average_mode: Optional[str] = "embeddings",
    grouping_keys: Optional[List[str]] = None,
    batch_size: int = 128,
    score_norm_method: Optional[str] = None,
    report_per_class_metrics: bool = True,
    right_as_classes: bool = True,
    use_image_data: bool = False,
) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame]:
    """
    Generic function to score the model's ability to produce similar embeddings for matching left and right inputs.
    Supports any combination of anndata, tensor, and list[str] for both left and right inputs.

    :param model: TranscriptomeTextDualEncoderModel instance. Both left and right embeddings will be computed using this model.
    :param left_input: anndata.AnnData, torch.tensor, or List[str]
        If anndata.AnnData, compute embeddings using the appropriate modality. If torch.tensor, use provided embeddings. If List[str], compute text embeddings.
    :param right_input: anndata.AnnData, torch.tensor, or List[str]
        If anndata.AnnData, compute embeddings using the appropriate modality. If torch.tensor, use provided embeddings. If List[str], compute text embeddings.
    :param correct_right_idx_per_left: A list with the index in right_input of the correct match for each sample in left_input.
    :param average_mode: "embeddings" or None.
        If "embeddings", first embed each sample, then average the embeddings. If None, don't average, report results at the single-sample level.
    :param grouping_keys: A list with group indicators (one for each sample in left_input). If average_mode is not None, this must be provided and will be used to split the left_input into groups that will be averaged separately.
          Will also be used to label the rows in the result dataframe. If None, just use numbers.
    :param batch_size: int. Model processing in batches (to avoid OOM)
    :param score_norm_method: "zscore", "softmax", "01norm" or None. How to normalize the logits
            (similarity scores). "zscore" will zscore the logits across all terms. "softmax" will apply softmax to the logits.
            "01norm" will normalize the logits to the range [0,1]. If None, don't normalize.
    :param report_per_class_metrics: bool. If True, report the performance metrics per class (i.e. per sample). If False, only report the macro average.
    :param right_as_classes: bool. If True, calculate the score using the right input as classes (default).
        If False, calculate the score using the left input as classes (can be useful for retrieval scoring).
        If False, average_mode must be None.
    :param use_image_data: bool. If True, and if an input is anndata.AnnData, use image data instead of transcriptome data.
    Returns:
        A tuple of:
         -  A dictionary containing precision, recall (at k=1,5,10,50), accuracy, f1, and rocauc.
            These metrics are reported using macro averaging (i.e. averaging over classes)
         -  None if report_per_class_metrics=False, else a dataframe with class labels as rows and performance metrics as columns.
    """
    if grouping_keys is None and average_mode is not None:
        raise ValueError("If average_mode is not None, grouping_keys must be provided.")

    # Keep the original (per-sample) grouping_keys before score_left_vs_right deduplicates them
    original_grouping_keys = grouping_keys

    # Get the scores.
    # grouping_keys will be updated (deduplicated and put into correct order) if averaging is used, else kept the same.
    scores, grouping_keys = score_left_vs_right(
        left_input=left_input,
        right_input=right_input,
        model=model,
        logit_scale=model.discriminator.temperature.exp(),
        average_mode=average_mode,
        grouping_keys=grouping_keys,
        batch_size=batch_size,
        score_norm_method=score_norm_method,
        use_image_data=use_image_data,
    )  # scores is a tensor of shape n_right * n_left (if average_mode is None), or n_right * n_left_groups (otherwise).
    # It is normalized column-wise, i.e. for each left sample, the mean of scores is 0 (if using zscore normalization)

    # Prepare the labels and compute the performance metrics
    return prepare_metrics_and_labels(
        scores=scores,
        left_input=left_input,
        right_input=right_input,
        correct_right_idx_per_left=correct_right_idx_per_left,
        average_mode=average_mode,
        grouping_keys=grouping_keys,
        original_grouping_keys=original_grouping_keys,
        right_as_classes=right_as_classes,
        report_per_class_metrics=report_per_class_metrics,
    )


def prepare_metrics_and_labels(
    scores: torch.Tensor,
    left_input: Union[anndata.AnnData, torch.Tensor, List[str]],
    right_input: Union[anndata.AnnData, torch.Tensor, List[str]],
    correct_right_idx_per_left: List[int],
    average_mode: Optional[str],
    grouping_keys: Optional[List[str]],
    right_as_classes: bool,
    report_per_class_metrics: bool,
    original_grouping_keys: Optional[List[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame]:
    """
    Prepare the labels and compute the performance metrics using torchmetrics.
    """

    # labels for the left input, and the true classes
    if grouping_keys is None:
        grouping_keys = [str(x) for x in range(scores.shape[1])]
        true_class_indices = correct_right_idx_per_left
    elif not right_as_classes:
        # the correct left index for every right input:
        true_class_indices = [
            correct_right_idx_per_left.index(x)
            for x in range(len(correct_right_idx_per_left))
        ]
    else:
        if average_mode is not None:
            # grouping_keys has been deduplicated/sorted by score_left_vs_right.
            # We need to look up the correct right index for each group using
            # the original (per-sample) grouping keys and correct_right_idx_per_left.
            original_keys = original_grouping_keys or grouping_keys
            # Build a mapping from group name -> correct right index
            group_to_right_idx = {}
            for key, right_idx in zip(original_keys, correct_right_idx_per_left):
                group_to_right_idx[key] = right_idx
            true_class_indices = [
                group_to_right_idx[x] for x in grouping_keys
            ]
        else:
            # we can just use the annotations and true classes as they are
            true_class_indices = correct_right_idx_per_left

    # labels for the right input
    if type(right_input) == torch.Tensor:
        right_annotations = [str(x) for x in range(scores.shape[0])]
    elif isinstance(right_input, list):
        right_annotations = right_input
    else:
        right_annotations = [str(x) for x in range(scores.shape[0])]

    right_labels = [f"right: {x}" for x in right_annotations]
    left_labels = [f"left: {x}" for x in grouping_keys]

    if right_as_classes:
        scores = (
            scores.t()
        )  # Here, it's better to have samples=left, classes=right, so we transpose
        columns = right_labels
        index = left_labels
        num_classes = len(right_annotations)

    else:
        if average_mode is not None:
            raise ValueError(
                "If right_as_classes is False, average_mode must be None as well"
            )
        columns = left_labels
        index = right_labels
        num_classes = len(grouping_keys)  # TODO not sure if this is correct

    # Create a dataframe with the scores
    scores_df = pd.DataFrame(
        scores.float().cpu().numpy(),
        columns=columns,
        index=index,
    )
    if average_mode is not None and type(right_input) != torch.Tensor:
        scores_df = scores_df[sorted(scores_df.columns)]

    torchmetric_kwargs = {
        "preds": scores,  # scores.t() if right_as_classes else scores,  # TODO delete comment
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
            res_metrics[f"recall_at_{k}"] = (
                torchmetrics.functional.classification.multiclass_recall(
                    **torchmetric_kwargs
                ).detach()
            )
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
            index=right_annotations,
        )
        per_class_df.index.name = "class"
        per_class_df["n_samples_in_class"] = [
            true_class_indices.count(class_idx)
            for class_idx, _ in enumerate(right_annotations)
        ]

        confusion_df = pd.DataFrame(
            confusion_mtx.numpy(),
            index=right_annotations,
            columns=[f"n_samples_predicted_as_{x}" for x in right_annotations],
        )
        confusion_df.index.name = "class"
        per_class_df = pd.concat([per_class_df, confusion_df], axis=1)

    else:
        per_class_df = None

    return (
        macro_average_results,
        per_class_df,
    )


def get_performance_metrics_transcriptome_vs_text(
    model: TranscriptomeTextDualEncoderModel,
    modality_input: Union[anndata.AnnData, torch.Tensor],
    text_list_or_text_embeds: Union[List[str], torch.Tensor],
    correct_text_idx_per_transcriptome: List[int],
    average_mode: Optional[str] = "embeddings",
    grouping_keys: Optional[List[str]] = None,
    batch_size: int = 128,
    score_norm_method: Optional[str] = None,
    report_per_class_metrics: bool = True,
    text_as_classes: bool = True,
    use_image_data: bool = False,
) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame]:
    """
    Backward compatibility wrapper for get_performance_metrics_left_vs_right.
    """
    return get_performance_metrics_left_vs_right(
        model=model,
        left_input=modality_input,
        right_input=text_list_or_text_embeds,
        correct_right_idx_per_left=correct_text_idx_per_transcriptome,
        average_mode=average_mode,
        grouping_keys=grouping_keys,
        batch_size=batch_size,
        score_norm_method=score_norm_method,
        report_per_class_metrics=report_per_class_metrics,
        right_as_classes=text_as_classes,
        use_image_data=use_image_data,
    )
