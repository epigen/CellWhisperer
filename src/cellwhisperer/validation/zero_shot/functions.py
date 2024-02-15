import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
from cellwhisperer.utils.inference import score_transcriptomes_vs_texts
from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel
from cellwhisperer.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from cellwhisperer.jointemb.scgpt_model import ScGPTTranscriptomeProcessor
from transformers import AutoTokenizer
import anndata
import json
from typing import Tuple, Union, List, Dict, Optional
import torchmetrics
from copy import copy


def get_performance_metrics_transcriptome_vs_text(
    model: TranscriptomeTextDualEncoderModel,
    transcriptome_input: Union[anndata.AnnData, torch.Tensor],
    text_list_or_text_embeds: Union[List[str], torch.Tensor],
    correct_text_idx_per_transcriptome: List[int],
    average_mode: Optional[str] = "embeddings",
    grouping_keys: Optional[List[str]] = None,
    text_tokenizer: Optional[AutoTokenizer] = None,
    transcriptome_processor: Optional[
        Union[GeneformerTranscriptomeProcessor, ScGPTTranscriptomeProcessor]
    ] = None,
    batch_size: int = 128,
    score_norm_method: Optional[str] = "zscore",
    report_per_class_metrics: bool = True,
) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame]:
    """
    Score the model's ability to produce similar embeddings for the given matching texts and adata objects.
    :param model: TranscriptomeTextDualEncoderModel instance. Both transcriptome and text embeddings will be computed using this model.
    :param transcriptome_input: anndata.AnnData or torch.tensor (n_cells*embedding_size) \
        If anndata.AnnData, first compute the transcriptome embeddings. If torch.tensor, use the provided transcriptome embeddings.
    :param text_list_or_text_embeds: List[str] or torch.tensor (n_celltype * embedding_size). If List[str], compute the text embeddings for each text. \
        If torch.tensor, use the provided text embeddings.
    :param correct_text_idx_per_transcriptome: A list with the index in text_list_or_text_embeds of the correct text for each transcriptome in transcriptome_input.
    :param average_mode: "cells" or "embeddings" or None. If "cells", first average the transcriptome data across all cells of same celltype, then tokenize and embed. \
        If "embeddings", first tokenize and embed each cell, then average the embeddings. If None, don't average, report results at the single-transcriptome level. TODO "cells" does not work yet.
    :param grouping_keys: A list with group indicators (one for each transcriptome in transcriptome_input). If average_mode is not None, this must be provided and will be used to split the transciptome_input into groups that will be averaged separately.
          Will also be used to label the rows in the result dataframe. If None, just use numbers. 
    :param text_tokenizer: AutoTokenizer instance. Used to tokenize the text. Can be None if text_list_or_text_embeds is a torch.tensor.
    :param transcriptome_processor: GeneformerTranscriptomeProcessor or ScGPTTranscriptomeProcessor instance. Used to prepare/tokenize the transcriptome. Can be None if transcriptome_input is a torch.tensor.
    :param batch_size: int. Model processing in batches (to avoid OOM)
    :param score_norm_method: "zscore", "softmax", "01norm" or None. TODO - unclear what is best. How to normalize the logits \
            (similarity to the transcriptome). "zscore" will zscore the logits across all terms. "softmax" will apply softmax to the logits.\
            "01norm" will normalize the logits to the range [0,1]. If None, don't normalize.
    :param report_per_class_metrics: bool. If True, report the performance metrics per class (i.e. per transcriptome). If False, only report the macro average.
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
        model=model,
        transcriptome_input=transcriptome_input,
        text_list_or_text_embeds=text_list_or_text_embeds,
        average_mode=average_mode,
        grouping_keys=grouping_keys,
        transcriptome_processor=transcriptome_processor,
        text_tokenizer=text_tokenizer,
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
            true_classe_indices = correct_text_idx_per_transcriptome
    else:
        if average_mode is not None:
            true_classe_indices = [
                correct_text_idx_per_transcriptome[full_grouping_keys.index(x)]
                for x in grouping_keys
            ]  # If we are averaging, we need to subset the true classes to match the averaged transcriptomes
        else:
            # we can just use the annotations and true classes as they are
            true_classe_indices = correct_text_idx_per_transcriptome

    # labels for the text
    if type(text_list_or_text_embeds) == torch.Tensor:
        text_annotations = [str(x) for x in range(scores.shape[0])]
    else:
        text_annotations = text_list_or_text_embeds
    scores = (
        scores.t()
    )  # Here, it's better to have samples=transcriptomes, classes=texts, so we transpose

    # Create a dataframe with the scores
    scores_df = pd.DataFrame(
        scores.cpu().numpy(),
        columns=[f"text: {x}" for x in text_annotations],
        index=[f"transcriptome: {x}" for x in grouping_keys],
    )
    if average_mode is not None and type(text_list_or_text_embeds) != torch.Tensor:
        scores_df = scores_df[sorted(scores_df.columns)]

    num_classes = int(max(true_classe_indices) + 1)
    preds = scores
    target = torch.Tensor(true_classe_indices).long()

    torchmetric_kwargs = {
        "preds": preds,
        "target": target,
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
            true_classe_indices.count(class_idx)
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


def anndata_to_scored_keywords(
    transcriptome_input: Union[anndata.AnnData, torch.Tensor],
    model: Optional[TranscriptomeTextDualEncoderModel],
    terms: Union[str, Path, Dict],
    transcriptome_processor: Union[
        GeneformerTranscriptomeProcessor, ScGPTTranscriptomeProcessor
    ],
    text_tokenizer: AutoTokenizer,
    average_mode: str = "cells",
    batch_size: int = 64,
    additional_text_dict: dict = {},
    score_norm_method: Optional[str] = "zscore",
) -> pd.DataFrame:
    """
    Compute the similarity between transcriptome embeddings on the on hand and the EnrichR terms + cell metadata on the other hand. \
    TODO potential improvement: Creating the dataframe from the start would make the code simpler. 
    :param transcriptome_input: Either: anndata.AnnData instance, then all cells in the object will be used to compute a single transcriptome embedding. \
                  Or: torch.tensor instance (n_cells * embedding_dim), then the provided transcriptome embeddings will be used.
    :param model: TranscriptomeTextDualEncoderModel instance. Both transcriptome and text embeddings will be computed using this model.
    :param terms: Either a `Path` or `str` to the json file containing the biological terms to match transcriptomes against (e.g. Enrichr) or a dict containing such terms. Expected format in both cases: (keys: library name, values: list of terms)
    :param transcriptome_processor: GeneformerTranscriptomeProcessor or ScGPTTranscriptomeProcessor instance. Used to tokenize/prepare the transcriptome.
    :param text_tokenizer: AutoTokenizer instance. Used to tokenize the text.
    :param average_mode: "cells" or "embeddings". If "cells", first average the transcriptome data across all cells, then tokenize and embed. \
        If "embeddings", first tokenize and embed each cell, then average the embeddings. TODO what is better?
    :param batch_size: int. The text will be chunked into chunks of this size before computing the text \
          embeddings and similarity to the transcriptome. This is necessary to avoid out-of-memory errors.
    :param additional_text_dict: dict. Additional text to compute the similarity to the transcriptome for. \
        Will be embedded as '<key>: <value>'.\
          E.g. if additional_text_dict={"day_of_induction": ["10","20"]}, the similarity to the transcriptome will be computed for \
          "day_of_induction: 10" and "day_of_induction: 20". 
    :param score_norm_method: "zscore", "softmax", "01norm" or None. TODO - unclear what is best. How to normalize the logits \
            (similarity to the transcriptome). "zscore" will zscore the logits across all terms. "softmax" will apply softmax to the logits.\
            "01norm" will normalize the logits to the range [0,1]. If None, don't normalize.
    :return: pd.DataFrame with the normalized logits (similarity to the transcriptome) for each term.
    """

    # we don't allow average_mode=None here, because we always want to create a single embedding from all provided cells
    assert average_mode in [
        "cells",
        "embeddings",
    ], f"average_mode must be one of ['cells', 'embeddings'], but is {average_mode}"

    if type(transcriptome_input) == anndata.AnnData:
        # We give every cell the same annotation (selected_cells), so that will be averaged to the same embedding later
        grouping_keys = [
            "selected_cells" for _ in range(transcriptome_input.obs.shape[0])
        ]
    elif type(transcriptome_input) == torch.Tensor:
        # We give every cell the same annotation (selected_cells), so that will be averaged to the same embedding later
        grouping_keys = ["selected_cells" for _ in range(transcriptome_input.shape[0])]
    else:
        raise ValueError(
            f"transcriptome_input must be either an anndata.AnnData instance or a torch.tensor, but is {type(transcriptome_input)}"
        )

    assert score_norm_method in [
        "zscore",
        "softmax",
        "01norm",
        None,
    ], f"score_norm_method must be one of ['zscore', 'softmax', '01norm'], but is {score_norm_method}"

    if isinstance(terms, (str, Path)):
        assert os.path.exists(terms), f"terms json path {terms} does not exist"

        ### Prepare text ###

        # EnrichR terms
        with open(terms, "r") as f:
            terms = json.load(f)

    n_terms_per_lib = {lib: len(terms[lib]) for lib in terms.keys()}
    terms_list = [term for lib in terms.keys() for term in terms[lib]]  # 16366 terms

    # Add values in the provided obs columns to the text
    text = terms_list

    # Add additional text to the text
    for key, value in additional_text_dict.items():
        text += value
        terms[key] = value
        n_terms_per_lib[key] = len(terms[key])

    #### Get text embeddings and compare to transcriptome embeddings ####
    logging.info("Computing text embeddings and logits...")

    scores, _ = score_transcriptomes_vs_texts(
        model=model,
        transcriptome_input=transcriptome_input,
        text_list_or_text_embeds=text,
        average_mode=average_mode,
        grouping_keys=grouping_keys,
        text_tokenizer=text_tokenizer,
        transcriptome_processor=transcriptome_processor,
        batch_size=batch_size,
        score_norm_method=score_norm_method,
    )  # n_text * 1

    logits_per_text = scores.squeeze(
        dim=1
    )  # squeeze the dimension 1, which is 1 because we only have one transcriptome embedding. Now: dim = n_text

    # split text into the different libraries and obs columns, and rank by normalized logits
    logits_df = pd.DataFrame(logits_per_text.numpy(), index=text, columns=["logits"])
    logits_df["term_without_prefix"] = np.nan
    logits_df["term_without_prefix"] = logits_df["term_without_prefix"].astype("object")
    i = 0
    # TODO all of this might be easier
    for library in terms.keys():
        text_this_lib = text[i : i + n_terms_per_lib[library]]
        logits_df.loc[text_this_lib, "library"] = library
        logits_df.loc[text_this_lib, "term_without_prefix"] = logits_df.loc[
            text_this_lib
        ].index.str.replace(f"{library}: ", "")
        i += n_terms_per_lib[library]
    logits_df["rank_in_library"] = logits_df.groupby("library")["logits"].rank(
        ascending=False
    )
    logits_df["rank_total"] = logits_df["logits"].rank(ascending=False)

    return logits_df.sort_values(by="logits", ascending=False)


def formatted_text_from_df(df, n_top_per_term):
    """
    Format the output of anndata_to_scored_keywords() as a string.
    :param df: pd.DataFrame. Output of anndata_to_scored_keywords().
    :param n_top_per_term: int. The top n terms per library will be returned.
    :return: str. Formatted text.
    """
    top_n_per_split = []
    for library, group in df.groupby("library"):
        top_n_terms = group.sort_values(by="logits", ascending=False).head(
            n_top_per_term
        )
        top_n_text = "\n\t".join(
            [
                f"{row['term_without_prefix']} ({row['logits']:.2f})"
                for _, row in top_n_terms.iterrows()
            ]
        )
        top_n_text = f"{library}:\n\t{top_n_text}"
        top_n_per_split.append(top_n_text)

    return "\n".join(top_n_per_split)
