import os
from pathlib import Path
import logging
import numpy as np
from scipy import sparse, stats
import pandas as pd
import torch
from single_cellm.jointemb.model import TranscriptomeTextDualEncoderModel
from single_cellm.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from transformers import AutoTokenizer
import anndata
import json
from typing import Tuple, Union, List, Dict
import torchmetrics


def adata_list_to_embeds(
    adata_list: List[anndata.AnnData],
    model: TranscriptomeTextDualEncoderModel,
    transcriptome_processor: GeneformerTranscriptomeProcessor,
    device: torch.device,
) -> torch.tensor:
    """
    Compute the transcriptome embeddings for each adata in adata_list.
    TODO: Only works with same number of cells per adata

    :param adata_list: List[anndata.AnnData] instance. All cells in each adata will be used to compute a single transcriptome embedding.
    :param model: TranscriptomeTextDualEncoderModel instance. Used to compute the transcriptome embeddings.
    :param transcriptome_processor: GeneformerTranscriptomeProcessor instance. Used to tokenize the transcriptome.
    :param device: torch.device instance
    :return: torch.tensor of transcriptome embeddings. Shape: n_adatas * n_cells_per_adata * embedding_size (e.g. 512)
    """
    transcriptome_embeds_all_adata = None
    for adata in adata_list:
        transcriptome_tokens = transcriptome_processor(
            adata, return_tensors="pt", padding=True
        )
        # make sure transcriptome_tokens are on GPU
        # TODO: Prepare for the case when the transcriptome is too large to fit on the GPU
        for k, v in transcriptome_tokens.items():
            transcriptome_tokens[k] = v.to(device)

        _, transcriptome_embeds = model.get_transcriptome_features(
            **transcriptome_tokens
        )
        transcriptome_embeds = transcriptome_embeds / transcriptome_embeds.norm(
            dim=-1, keepdim=True
        )
        if transcriptome_embeds_all_adata is None:
            transcriptome_embeds_all_adata = transcriptome_embeds.unsqueeze(0)
        else:
            transcriptome_embeds_all_adata = torch.cat(
                [transcriptome_embeds_all_adata, transcriptome_embeds.unsqueeze(0)],
                dim=0,
            )
    return transcriptome_embeds_all_adata


def text_list_to_embeds(
    text_list: List[str],
    model: TranscriptomeTextDualEncoderModel,
    text_tokenizer: AutoTokenizer,
    device: torch.device,
) -> torch.tensor:
    """
    Compute the text embeddings for each text in text_list.
    :param text: List[str] instance. Each text will be tokenized and embedded.
    :param model: TranscriptomeTextDualEncoderModel instance. Used to compute the text embeddings.
    :param text_tokenizer: AutoTokenizer instance. Used to tokenize the text.
    :param device: torch.device instance
    :return: torch.tensor of text embeddings. Shape: len(text_list) * embedding_size (e.g. 512)
    """
    # Tokenize the chunk and move it to the device
    text_tokens = text_tokenizer(text_list, return_tensors="pt", padding=True)
    for k, v in text_tokens.items():
        text_tokens[k] = v.to(device)

    # Compute text embeddings
    _, text_embeds = model.get_text_features(**text_tokens)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    return text_embeds


def score_text_vs_transcriptome_many_vs_many(
    model: TranscriptomeTextDualEncoderModel,
    device: torch.device,
    adata_list_or_transcriptome_embeds: Union[List[anndata.AnnData], torch.tensor],
    text_list_or_text_embeds: Union[List[str], torch.tensor],
    average_mode: str = "embeddings",
    text_tokenizer: AutoTokenizer = None,
    transcriptome_processor: GeneformerTranscriptomeProcessor = None,
    chunk_size_text_emb_and_scoring: int = 128,
    score_norm_method: str = "zscore",
):
    """
    Compute the similarity between the text and the transcriptome embeddings.
    :param model: TranscriptomeTextDualEncoderModel instance. Both transcriptome and text embeddings will be computed using this model.
    :param device: torch.device instance
    :param adata_list_or_transcriptome_embeds: List[anndata.AnnData] or torch.tensor. If List[anndata.AnnData], compute the transcriptome embeddings for each adata. \
        If torch.tensor, use the provided transcriptome embeddings.
    :param text_list_or_text_embeds: List[str] or torch.tensor. If List[str], compute the text embeddings for each text. \
        If torch.tensor, use the provided text embeddings.
    :param average_mode: "cells" or "embeddings". If "cells", first average the transcriptome data across all cells, then tokenize and embed. \
        If "embeddings", first tokenize and embed each cell, then average the embeddings. TODO "cells" does not work yet.
    :param text_tokenizer: AutoTokenizer instance. Used to tokenize the text. Can be None if text_list_or_text_embeds is a torch.tensor.
    :param transcriptome_processor: GeneformerTranscriptomeProcessor instance. Used to tokenize the transcriptome. Can be None if adata_list_or_transcriptome_embeds is a torch.tensor.
    :param chunk_size_text_emb_and_scoring: int. The text will be chunked into chunks of this size before computing the text \
            embeddings and similarity to the transcriptome. This is necessary to avoid out-of-memory errors.
    :param score_norm_method: "zscore", "softmax", or "01norm". TODO - unclear what is best. How to normalize the logits \
            (similarity to the transcriptome). "zscore" will zscore the logits across all terms. "softmax" will apply softmax to the logits.\
            "01norm" will normalize the logits to the range [0,1].
    : return: torch.tensor of similarity between the text and the adatas. Shape: n_text * n_adata

    """
    logit_scale = model.logit_scale.exp()

    #### Prepare transcriptome embeddings ###
    if average_mode == "cells":
        raise NotImplementedError("average_mode='cells' not implemented yet")  # TODO
        # TypeError: sum() got an unexpected keyword argument 'keepdims' in transcriptome_processor when providing an anndata with only 1 cell
        # average_adata_list = [anndata.AnnData(
        #     expression.X.mean(axis=0), # TODO I get an error with keepdims=True),
        #     var=expression.var,
        # ) for expression in adata_list_or_transcriptome_embeds]
        # adata_list_or_transcriptome_embeds=average_adata_list
    if type(adata_list_or_transcriptome_embeds) == torch.Tensor:
        transcriptome_embeds = adata_list_or_transcriptome_embeds
    else:
        transcriptome_embeds = adata_list_to_embeds(
            adata_list_or_transcriptome_embeds, model, transcriptome_processor, device
        )  # n_adatas * n_cells_per_adata * 512
    if average_mode == "embeddings":
        transcriptome_embeds = transcriptome_embeds.mean(
            dim=1, keepdim=False
        )  # now: n_adatas * 512

    #### Chunk the text to avoid out-of-memory errors ###
    logits_per_text_list = []
    text_chunks = [
        text_list_or_text_embeds[i : i + chunk_size_text_emb_and_scoring]
        for i in range(
            0, len(text_list_or_text_embeds), chunk_size_text_emb_and_scoring
        )
    ]
    for chunk in text_chunks:
        if type(text_list_or_text_embeds) == torch.Tensor:
            text_embeds = chunk
        else:
            text_embeds = text_list_to_embeds(
                chunk, model, text_tokenizer, device
            )  # chunk_size_text_emb_and_scoring * 512

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

    return logits_per_text


def get_scores_adatas_vs_text_list(
    adata_dict_or_embedding_dict: Dict[str, Union[anndata.AnnData, torch.Tensor]],
    model: TranscriptomeTextDualEncoderModel,
    device: torch.device,
    text_tokenizer: AutoTokenizer,
    transcriptome_processor: GeneformerTranscriptomeProcessor,
    text_list_or_text_embeds: Union[List[str], torch.Tensor] = None,
    average_mode="embeddings",
    chunk_size_text_emb_and_scoring: int = 64,
    score_norm_method: str = "zscore",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Score the model's ability to produce similar embeddings for the given texts and adata objects.

    Args:
        adata_dict_or_embedding_dict: Either: A dictionary of anndata objects (each of a single cell type), \
              where the keys are the celltypes and the values are the anndata objects. \
                Or: A dictionary of transcriptome embedding tensors (each of a single cell type), \
                    where the keys are the celltypes and the values are the transcriptome embeddings. \
        model: The model to use for scoring.
        device: The device to use for scoring.
        text_tokenizer: The tokenizer to use for scoring.
        transcriptome_processor: The transcriptome processor to use for scoring.
        text_list_or_text_embeds: Either: A list of texts to score. Or: A tensor of text embeddings to score. \
            If None, the celltypes in adata_dict will be used, with the prefix "Cell type: ".
        average_mode: "cells" or "embeddings". If "cells", first average the transcriptome data across all cells, then tokenize and embed. \
            If "embeddings", first tokenize and embed each cell, then average the embeddings. TODO "cells" does not work yet.
        chunk_size_text_emb_and_scoring: The text will be chunked into chunks of this size before computing the text \
            embeddings and similarity to the transcriptome. This is necessary to avoid out-of-memory errors.
        score_norm_method: "zscore", "softmax", or "01norm". TODO - unclear what is best. How to normalize the logits \
            (similarity to the transcriptome). "zscore" will zscore the logits across all terms. "softmax" will apply softmax to the logits.\
                "01norm" will normalize the logits to the range [0,1].
    Returns:
        1. A dictionary with the following keys:
            precision: The precision score.
            recall: The recall score.
            accuracy: The accuracy score.
            f1: The f1 score.
        2. A dataframe with the similarity scores for each text and adata combination.
    """

    celltypes_to_process = adata_dict_or_embedding_dict.keys()
    adata_list_or_transcriptome_embeds = adata_dict_or_embedding_dict.values()

    if text_list_or_text_embeds is None:
        text_list_or_text_embeds = [f"Cell type: {x}" for x in celltypes_to_process]

    # Get the scores
    scores = score_text_vs_transcriptome_many_vs_many(
        model=model,
        device=device,
        text_list_or_text_embeds=text_list_or_text_embeds,
        adata_list_or_transcriptome_embeds=adata_list_or_transcriptome_embeds,
        transcriptome_processor=transcriptome_processor,
        text_tokenizer=text_tokenizer,
        chunk_size_text_emb_and_scoring=chunk_size_text_emb_and_scoring,
        score_norm_method=score_norm_method,
        average_mode=average_mode,
    )

    # scores is a tensor of shape n_text * n_adata
    # It is normalized column-wise, i.e. for each adata, the mean of scores is 0 (if using zscore normalization)

    # TODO also return unnormalized scores to show if there are adatas where none of the texts are close in embedding space?

    scores_df = pd.DataFrame(
        scores.cpu().numpy(),
        index=[f"prediction: {x}" for x in celltypes_to_process],
        columns=[f"input adata: {x}" for x in celltypes_to_process],
    )

    true_celltype = np.arange(len(celltypes_to_process))
    num_classes = int(max(true_celltype) + 1)
    preds = (
        scores.t()
    )  # torchmetrics expects preds to be of shape n_samples * n_classes. Our "samples" are our adatas, so we need to transpose
    target = torch.Tensor(true_celltype).long()
    torchmetric_kwargs = {
        "preds": preds,
        "target": target,
        "num_classes": num_classes,
        "average": "macro",
        "top_k": 1,
    }

    precision = torchmetrics.functional.classification.multiclass_precision(
        **torchmetric_kwargs
    )
    recall = torchmetrics.functional.classification.multiclass_recall(
        **torchmetric_kwargs
    )
    accuracy = torchmetrics.functional.classification.multiclass_accuracy(
        **torchmetric_kwargs
    )
    f1 = torchmetrics.functional.classification.multiclass_f1_score(
        **torchmetric_kwargs
    )

    result_dict = {
        "precision": precision.item(),
        "recall": recall.item(),
        "accuracy": accuracy.item(),
        "f1": f1.item(),
    }

    return result_dict, scores_df


def anndata_to_scored_keywords(
    adata_or_embedding: Union[anndata.AnnData, torch.Tensor],
    model: TranscriptomeTextDualEncoderModel,
    terms_json_path: Union[str, Path],
    transcriptome_processor: GeneformerTranscriptomeProcessor,
    text_tokenizer: AutoTokenizer,
    device: torch.device,
    average_mode: str = "cells",
    chunk_size_text_emb_and_scoring: int = 64,
    obs_cols: List[str] = [],
    additional_text_dict: dict = {},
    score_norm_method: str = "zscore",
) -> Union[pd.DataFrame, str]:
    """
    Compute the similarity between transcriptome embeddings on the on hand and the EnrichR terms + cell metadata on the other hand. \
    TODO potential improvement: Creating the dataframe from the start would make the code simpler. 
    :param adata_or_embedding: Either: anndata.AnnData instance, then all cells in the object will be used to compute a single transcriptome embedding. \
                  Or: torch.tensor instance, then the provided transcriptome embeddings will be used.
    :param model: TranscriptomeTextDualEncoderModel instance. Both transcriptome and text embeddings will be computed using this model.
    :param terms_json_path: Path to the json file containing the EnrichR terms (keys: libraries, values: list of terms)
    :param transcriptome_processor: GeneformerTranscriptomeProcessor instance. Used to tokenize the transcriptome.
    :param text_tokenizer: AutoTokenizer instance. Used to tokenize the text.
    :param device: torch.device instance
    :param average_mode: "cells" or "embeddings". If "cells", first average the transcriptome data across all cells, then tokenize and embed. \
        If "embeddings", first tokenize and embed each cell, then average the embeddings. TODO what is better?
    :param chunk_size_text_emb_and_scoring: int. The text will be chunked into chunks of this size before computing the text \
          embeddings and similarity to the transcriptome. This is necessary to avoid out-of-memory errors.
    :param obs_cols: List[str]. Compute the similarity to the transcriptome for the values of these columns.\
          E.g. if obs_cols=["cell type"], the similarity to the transcriptome will be computed for each value of "cell type". \
        Note that the column name will be prepended to each value before the embedding, e.g. "cell type: B cell". \
            Therefore, columns should be informatively named..
    :param additional_text_dict: dict. Additional text to compute the similarity to the transcriptome for. \
        Will be embedded as '<key>: <value>'.\
          E.g. if additional_text_dict={"day_of_induction": ["10","20"]}, the similarity to the transcriptome will be computed for \
          "day_of_induction: 10" and "day_of_induction: 20". 
    :param score_norm_method: "zscore", "softmax", or "01norm". TODO - unclear what is best. How to normalize the logits \
        (similarity to the transcriptome). "zscore" will zscore the logits across all terms. "softmax" will apply softmax to the logits.\
              "01norm" will normalize the logits to the range [0,1].
    :return: pd.DataFrame with the normalized logits (similarity to the transcriptome) for each term.
    """

    assert average_mode in [
        "cells",
        "embeddings",
    ], f"average_mode must be one of ['cells', 'embeddings'], but is {average_mode}"

    if type(adata_or_embedding) == anndata.AnnData:
        expression = adata_or_embedding
        assert all(
            [x in expression.obs.columns for x in obs_cols]
        ), f"obs_cols must be a subset of {expression.obs.columns}, but is {obs_cols}"
    elif type(adata_or_embedding) == torch.Tensor:
        assert (
            obs_cols == []
        ), f"obs_cols must be empty if adata_dict_or_embedding is a tensor, but is {obs_cols}"
    assert os.path.exists(
        terms_json_path
    ), f"terms_json_path {terms_json_path} does not exist"
    assert score_norm_method in [
        "zscore",
        "softmax",
        "01norm",
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
            f"{obs_col}: {value}" for value in expression.obs[obs_col].unique().tolist()
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

    scores = score_text_vs_transcriptome_many_vs_many(
        model=model,
        device=device,
        text_list_or_text_embeds=text,
        adata_list_or_transcriptome_embeds=[expression]
        if type(adata_or_embedding) == anndata.AnnData
        else adata_or_embedding,
        transcriptome_processor=transcriptome_processor,
        text_tokenizer=text_tokenizer,
        chunk_size_text_emb_and_scoring=chunk_size_text_emb_and_scoring,
        score_norm_method=score_norm_method,
        average_mode=average_mode,
    )

    logits_per_text = scores.squeeze(
        dim=1
    )  # squeeze the first dimension, which is 1 because we only have one transcriptome embedding

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
