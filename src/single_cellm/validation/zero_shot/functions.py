import os
from pathlib import Path
import logging
import numpy as np
from scipy import sparse, stats
import pandas as pd
import torch
from single_cellm.jointemb.model import TranscriptomeTextDualEncoderModel
from single_cellm.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from single_cellm.jointemb.scgpt_model import ScGPTTranscriptomeProcessor
from transformers import AutoTokenizer
import anndata
import json
from typing import Tuple, Union, List, Dict, Optional
import torchmetrics
from copy import copy


def adata_to_embeds(
    adata: anndata.AnnData,
    model: TranscriptomeTextDualEncoderModel,
    transcriptome_processor: Union[
        GeneformerTranscriptomeProcessor, ScGPTTranscriptomeProcessor
    ],
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Compute the transcriptome embeddings for each cell in the adata object.
    :param adata: anndata.AnnData instance.
    :param model: TranscriptomeTextDualEncoderModel instance. Used to compute the transcriptome embeddings.
    :param transcriptome_processor: GeneformerTranscriptomeProcessor or ScGPTTranscriptomeProcessor instance. Used to prepare/tokenize the transcriptome.
    :return: torch.tensor of transcriptome embeddings. Shape: n_transcriptomes_in_adata * embedding_size (e.g. 512)
    """

    transcriptome_processor_result = transcriptome_processor(
        adata, return_tensors="pt", padding=True
    )
    # make sure transcriptome_tokens are on GPU
    # TODO: Prepare for the case when the transcriptome is too large to fit on the GPU
    for k, v in transcriptome_processor_result.items():
        transcriptome_processor_result[k] = v.to(model.device)

    transcriptome_embeds = []
    for i in range(
        0,
        next(iter(transcriptome_processor_result.values())).shape[0],
        batch_size,
    ):
        batch = {
            k: v[i : i + batch_size] for k, v in transcriptome_processor_result.items()
        }
        _, transcriptome_embeds_batch = model.get_transcriptome_features(**batch)
        transcriptome_embeds.append(transcriptome_embeds_batch)
    transcriptome_embeds = torch.cat(transcriptome_embeds, dim=0)

    transcriptome_embeds = transcriptome_embeds / transcriptome_embeds.norm(
        dim=-1, keepdim=True
    )

    return transcriptome_embeds


def text_list_to_embeds(
    text_list: List[str],
    model: TranscriptomeTextDualEncoderModel,
    text_tokenizer: AutoTokenizer,
) -> torch.tensor:
    """
    Compute the text embeddings for each text in text_list.
    :param text: List[str] instance. Each text will be tokenized and embedded.
    :param model: TranscriptomeTextDualEncoderModel instance. Used to compute the text embeddings.
    :param text_tokenizer: AutoTokenizer instance. Used to tokenize the text.
    :return: torch.tensor of text embeddings. Shape: len(text_list) * embedding_size (e.g. 512)
    """
    # Tokenize the chunk and move it to the device
    text_tokens = text_tokenizer(text_list, return_tensors="pt", padding=True)
    for k, v in text_tokens.items():
        text_tokens[k] = v.to(model.device)

    # Compute text embeddings
    _, text_embeds = model.get_text_features(**text_tokens)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    return text_embeds


def score_text_vs_transcriptome_many_vs_many(
    model: TranscriptomeTextDualEncoderModel,
    transcriptome_input: Union[anndata.AnnData, torch.tensor],
    text_list_or_text_embeds: Union[List[str], torch.tensor],
    average_mode: Optional[str] = "embeddings",
    transcriptome_annotations: Optional[List[str]] = None,
    text_tokenizer: Optional[AutoTokenizer] = None,
    transcriptome_processor: Optional[
        Union[GeneformerTranscriptomeProcessor, ScGPTTranscriptomeProcessor]
    ] = None,
    batch_size: int = 128,
    score_norm_method: Optional[str] = "zscore",
) -> Tuple[torch.tensor, torch.tensor, Optional[List[str]]]:
    """
    Compute the similarity between the text and the transcriptome embeddings.
    :param model: TranscriptomeTextDualEncoderModel instance. Both transcriptome and text embeddings will be computed using this model.
    :param transcriptome_input: anndata.AnnData or torch.tensor (n_cells*embedding_size) \
        If anndata.AnnData, first compute the transcriptome embeddings. If torch.tensor, use the provided transcriptome embeddings.
    :param text_list_or_text_embeds: List[str] or torch.tensor (n_celltype * embedding_size). If List[str], compute the text embeddings for each text. \
        If torch.tensor, use the provided text embeddings.
    :param average_mode: "cells" or "embeddings" or None. If "cells", first average the transcriptome data across all cells of same celltype, then tokenize and embed. \
        If "embeddings", first tokenize and embed each cell, then average the embeddings. If None, don't average, report scores at the single-transcriptome level. TODO "cells" does not work yet.
    :param transcriptome_annotations: A list with labels for each transcriptome in transcriptome_input. If average_mode is not None, this must be provided and will be used to split the transciptome_input into pieces that will be averaged separately.
    :param text_tokenizer: AutoTokenizer instance. Used to tokenize the text. Can be None if text_list_or_text_embeds is a torch.tensor.
    :param transcriptome_processor: GeneformerTranscriptomeProcessor or ScGPTTranscriptomeProcessor instance. Used to prepare/tokenize the transcriptome. Can be None if transcriptome_input is a torch.tensor.
    :param batch_size: int. Model processing in batches (to avoid OOM)
    :param score_norm_method: "zscore", "softmax", "01norm" or None. TODO - unclear what is best. How to normalize the logits \
            (similarity to the transcriptome). "zscore" will zscore the logits across all terms. "softmax" will apply softmax to the logits.\
            "01norm" will normalize the logits to the range [0,1]. If None, don't normalize.
    : return: A tuple of three objects: \
        First, a torch.tensor of  similarity between the text and the adatas with shape: n_text * n_adata. \
        second, the list of transcriptome annotations corresponding to the dim1 of the tensors (or None if transcriptome_annotations is None).
    """
    logit_scale = model.discriminator.temperature.exp()

    # check inputs
    if type(transcriptome_input) == torch.Tensor:
        assert (
            transcriptome_input.dim() == 2
        ), f"transcriptome_input must be a tensor of shape n_cells * embedding_size, but is {transcriptome_input.shape}"
    assert not (
        transcriptome_annotations is None and average_mode is not None
    ), "transcriptome_annotations must be provided if average_mode is not None"

    #### Prepare transcriptome embeddings ###
    if average_mode == "cells":
        raise NotImplementedError("average_mode='cells' not implemented yet")  # TODO

    if type(transcriptome_input) == torch.Tensor:
        transcriptome_embeds = transcriptome_input
    else:
        transcriptome_embeds = adata_to_embeds(
            transcriptome_input,
            model,
            transcriptome_processor,
            batch_size=batch_size,
        )  # n_cells * 512

    if average_mode == "embeddings":
        sorted_unique_annotations = sorted(list(set(transcriptome_annotations)))
        averaged_transcriptome_embeds = []
        for annotation in sorted_unique_annotations:
            avg_emb_this_celltype = transcriptome_embeds[
                [annotation == x for x in transcriptome_annotations]
            ].mean(
                dim=0, keepdim=False
            )  # 512
            averaged_transcriptome_embeds.append(avg_emb_this_celltype)
        transcriptome_embeds = torch.stack(
            averaged_transcriptome_embeds
        )  # n_celltypes * 512
        transcriptome_annotations = sorted_unique_annotations

    #### Chunk the text to avoid out-of-memory errors ###
    logits_per_text_list = []
    text_chunks = [
        text_list_or_text_embeds[i : i + batch_size]
        for i in range(0, len(text_list_or_text_embeds), batch_size)
    ]
    for chunk in text_chunks:
        if type(text_list_or_text_embeds) == torch.Tensor:
            text_embeds = chunk
        else:
            text_embeds = text_list_to_embeds(
                chunk, model, text_tokenizer
            )  # batch_size * 512

        # Compute logits (similarity to expression embedding) for the current chunk and append to the list
        logits_per_text = (
            torch.matmul(text_embeds, transcriptome_embeds.t()) * logit_scale
        )  # n_text * n_adatas
        logits_per_text_list.append(logits_per_text.cpu().detach())

    # Concatenate the results to get the final text_embeds
    logits_per_text = torch.cat(logits_per_text_list, dim=0)

    # TODO: What is the best normalization here? Softmax, zscore, [0,1]? Something else?
    if score_norm_method == "softmax":
        logits_per_text = torch.softmax(logits_per_text, dim=0)
    elif score_norm_method == "01norm":
        logits_per_text = (logits_per_text - logits_per_text.min()) / (
            logits_per_text.max() - logits_per_text.min()
        )
    elif score_norm_method == "zscore":
        logits_per_text = torch.tensor(
            stats.zscore(logits_per_text.numpy(), axis=0), dtype=torch.float32
        )
    elif score_norm_method is None:
        pass
    else:
        raise ValueError(
            f"score_norm_method must be one of 'softmax', 'zscore', '01norm', or None, but is {score_norm_method}"
        )

    return (
        logits_per_text,  # n_text * n_cells, or n_text * n_celltypes
        transcriptome_annotations,  # n_cells or n_celltypes
    )


def get_performance_metrics_transcriptome_vs_text(
    model: TranscriptomeTextDualEncoderModel,
    transcriptome_input: Union[anndata.AnnData, torch.tensor],
    text_list_or_text_embeds: Union[List[str], torch.tensor],
    correct_text_idx_per_transcriptome: List[int],
    average_mode: Optional[str] = "embeddings",
    transcriptome_annotations: Optional[List[str]] = None,
    text_tokenizer: Optional[AutoTokenizer] = None,
    transcriptome_processor: Optional[
        Union[GeneformerTranscriptomeProcessor, ScGPTTranscriptomeProcessor]
    ] = None,
    batch_size: int = 128,
    score_norm_method: Optional[str] = "zscore",
    report_per_class_metrics: bool = True,
) -> Tuple[Union[Dict[str, torch.tensor], pd.DataFrame]]:
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
    :param transcriptome_annotations:  A list with labels for each transcriptome in transcriptome_input. If average_mode is not None, this must be provided and will be used to split the transciptome_input into pieces that will be averaged separately. \
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

    full_transcriptome_annotations = copy(
        transcriptome_annotations
    )  # We keep the full annotations in case they get modified due to averaging below

    # Get the scores.
    # transcriptome_annotations will be updated (deduplicated and put into correct order) if averaging is used, else kept the same.
    scores, transcriptome_annotations = score_text_vs_transcriptome_many_vs_many(
        model=model,
        transcriptome_input=transcriptome_input,
        text_list_or_text_embeds=text_list_or_text_embeds,
        average_mode=average_mode,
        transcriptome_annotations=transcriptome_annotations,
        transcriptome_processor=transcriptome_processor,
        text_tokenizer=text_tokenizer,
        batch_size=batch_size,
        score_norm_method=score_norm_method,
    )  # scores is a tensor of shape n_text * n_cells (if average_mode is None), or n_text * n_celltypes (otherwise).
    # It is normalized column-wise, i.e. for each adata, the mean of scores is 0 (if using zscore normalization)

    # labels for the transcriptome, and the true classes
    if transcriptome_annotations is None:
        transcriptome_annotations = [str(x) for x in range(scores.shape[1])]
        if average_mode is not None:
            raise ValueError(
                "If transcriptome_annotations is None, average_mode must be None as well"
            )
        else:
            true_classe_indices = correct_text_idx_per_transcriptome
    else:
        if average_mode is not None:
            true_classe_indices = [
                correct_text_idx_per_transcriptome[
                    full_transcriptome_annotations.index(x)
                ]
                for x in transcriptome_annotations
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
        index=[f"transcriptome: {x}" for x in transcriptome_annotations],
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
    model: TranscriptomeTextDualEncoderModel,
    terms_json_path: Union[str, Path],
    transcriptome_processor: Union[
        GeneformerTranscriptomeProcessor, ScGPTTranscriptomeProcessor
    ],
    text_tokenizer: AutoTokenizer,
    average_mode: str = "cells",
    batch_size: int = 64,
    obs_cols: List[str] = [],
    additional_text_dict: dict = {},
    score_norm_method: Optional[str] = "zscore",
) -> Union[pd.DataFrame, str]:
    """
    Compute the similarity between transcriptome embeddings on the on hand and the EnrichR terms + cell metadata on the other hand. \
    TODO potential improvement: Creating the dataframe from the start would make the code simpler. 
    :param transcriptome_input: Either: anndata.AnnData instance, then all cells in the object will be used to compute a single transcriptome embedding. \
                  Or: torch.tensor instance (n_cells * embedding_dim), then the provided transcriptome embeddings will be used.
    :param model: TranscriptomeTextDualEncoderModel instance. Both transcriptome and text embeddings will be computed using this model.
    :param terms_json_path: Path to the json file containing the EnrichR terms (keys: libraries, values: list of terms)
    :param transcriptome_processor: GeneformerTranscriptomeProcessor or ScGPTTranscriptomeProcessor instance. Used to tokenize/prepare the transcriptome.
    :param text_tokenizer: AutoTokenizer instance. Used to tokenize the text.
    :param average_mode: "cells" or "embeddings". If "cells", first average the transcriptome data across all cells, then tokenize and embed. \
        If "embeddings", first tokenize and embed each cell, then average the embeddings. TODO what is better?
    :param batch_size: int. The text will be chunked into chunks of this size before computing the text \
          embeddings and similarity to the transcriptome. This is necessary to avoid out-of-memory errors.
    :param obs_cols: List[str]. Compute the similarity to the transcriptome for the values of these columns.\
          E.g. if obs_cols=["cell type"], the similarity to the transcriptome will be computed for each value of "cell type". \
        Note that the column name will be prepended to each value before the embedding, e.g. "cell type: B cell". \
            Therefore, columns should be informatively named..
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
        assert all(
            [x in transcriptome_input.obs.columns for x in obs_cols]
        ), f"obs_cols must be a subset of {transcriptome_input.obs.columns}, but is {obs_cols}"
        # We give every cell the same annotation (selected_cells), so that will be averaged to the same embedding later
        transcriptome_annotations = [
            "selected_cells" for _ in range(transcriptome_input.obs.shape[0])
        ]
    elif type(transcriptome_input) == torch.Tensor:
        assert (
            obs_cols == []
        ), f"obs_cols must be empty if transcriptome_input is a tensor, but is {obs_cols}"
        # We give every cell the same annotation (selected_cells), so that will be averaged to the same embedding later
        transcriptome_annotations = [
            "selected_cells" for _ in range(transcriptome_input.shape[0])
        ]

    assert os.path.exists(
        terms_json_path
    ), f"terms_json_path {terms_json_path} does not exist"
    assert score_norm_method in [
        "zscore",
        "softmax",
        "01norm",
        None,
    ], f"score_norm_method must be one of ['zscore', 'softmax', '01norm'], but is {score_norm_method}"

    ### Prepare text ###

    # EnrichR terms
    with open(terms_json_path, "r") as f:
        terms = json.load(f)

    n_terms_per_lib = {lib: len(terms[lib]) for lib in terms.keys()}
    terms_list = [term for lib in terms.keys() for term in terms[lib]]  # 16366 terms

    # Add values in the provided obs columns to the text
    text = terms_list
    for obs_col in obs_cols:
        text_this_obs_col = [
            f"{obs_col}: {value}"
            for value in transcriptome_input.obs[obs_col].unique().tolist()
        ]
        text += text_this_obs_col
        terms[obs_col] = text_this_obs_col
        n_terms_per_lib[obs_col] = len(terms[obs_col])

    # Add additional text to the text
    for key, value in additional_text_dict.items():
        text_this_key = [f"{key}: {v}" for v in value]
        text += text_this_key
        terms[key] = text_this_key
        n_terms_per_lib[key] = len(terms[key])

    #### Get text embeddings and compare to transcriptome embeddings ####
    logging.info("Computing text embeddings and logits...")

    scores, _ = score_text_vs_transcriptome_many_vs_many(
        model=model,
        transcriptome_input=transcriptome_input,
        text_list_or_text_embeds=text,
        average_mode=average_mode,
        transcriptome_annotations=transcriptome_annotations,
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
