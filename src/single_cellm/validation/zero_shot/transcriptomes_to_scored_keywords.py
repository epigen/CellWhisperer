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
from typing import Union, List


def anndata_to_scored_keywords(
    expression: anndata.AnnData,
    model: TranscriptomeTextDualEncoderModel,
    terms_json_path: Union[str, Path],
    transcriptome_processor: GeneformerTranscriptomeProcessor,
    text_tokenizer: AutoTokenizer,
    device: torch.device,
    average_mode: str = "cells",
    chunk_size_text_emb_and_scoring: int = 64,
    obs_cols: List[str] = ["cell type", "cell type rough"],
    additional_text_dict: dict = {},
    score_norm_method: str = "zscore",
    return_mode: str = "DataFrame",
) -> Union[pd.DataFrame, str]:
    """
    Compute the similarity between transcriptome embeddings on the on hand and the EnrichR terms + cell metadata on the other hand. \
    TODO potential improvement: Creating the dataframe from the start would make the code simpler. 
    :param adata: anndata.AnnData instance. All cells in the object will be used to compute a single transcriptome embedding.
    :param model: TranscriptomeTextDualEncoderModel instance. Both transcriptome and text embeddings will be computed using this model.
    :param terms_json_path: Path to the json file containing the EnrichR terms (keys: libraries, values: list of terms)
    :param transcriptome_processor: GeneformerTranscriptomeProcessor instance. Used to tokenize the transcriptome.
    :param text_tokenizer: AutoTokenizer instance. Used to tokenize the text.
    :param device: torch.device instance
    :param average_mode: "cells" or "embeddings". If "cells", first average the transcriptome data across all cells, then tokenize and embed. \
        If "embeddings", first tokenize and embed each cell, then average the embeddings. TODO what is better?
    :param chunk_size_text_emb_and_scoring: int. The text will be chunked into chunks of this size before computing the text \
          embeddings and similarity to the transcriptome. This is necessary to avoid out-of-memory errors.
    :param n_top_per_term: int. If return_mode is 'text' or 'dict', the top n terms per library will be returned.
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
    assert all(
        [x in expression.obs.columns for x in obs_cols]
    ), f"obs_cols must be a subset of {expression.obs.columns}, but is {obs_cols}"
    assert os.path.exists(
        terms_json_path
    ), f"terms_json_path {terms_json_path} does not exist"
    assert score_norm_method in [
        "zscore",
        "softmax",
        "01norm",
    ], f"score_norm_method must be one of ['zscore', 'softmax', '01norm'], but is {score_norm_method}"
    assert return_mode in [
        "DataFrame",
        "text",
    ], f"return_mode must be one of ['DataFrame', 'text','dict'], but is {return_mode}"

    #### Transcriptome embedding ####
    obs = expression.obs
    if average_mode == "cells":
        expression = anndata.AnnData(
            expression.X.mean(axis=0, keepdims=True),
            var=expression.var,
        )
    transcriptome_tokens = transcriptome_processor(
        expression, return_tensors="pt", padding=True
    )
    # make sure transcriptome_tokens are on GPU
    # TODO: Prepare for the case when the transcriptome is too large to fit on the GPU
    for k, v in transcriptome_tokens.items():
        transcriptome_tokens[k] = v.to(device)

    _, transcriptome_embeds = model.get_transcriptome_features(**transcriptome_tokens)
    transcriptome_embeds = transcriptome_embeds / transcriptome_embeds.norm(
        dim=-1, keepdim=True
    )
    if average_mode == "embeddings":
        transcriptome_embeds = transcriptome_embeds.mean(dim=0, keepdim=True)

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
            f"{obs_col}: {value}" for value in obs[obs_col].unique().tolist()
        ]
        text += text_this_obs_col
        terms[obs_col] = text_this_obs_col
        n_terms_per_lib[obs_col] = len(terms[obs_col])

    for key, value in additional_text_dict.items():
        text_this_key = [f"{key}: {v}" for v in value]
        text += text_this_key
        terms[key] = text_this_key
        n_terms_per_lib[key] = len(terms[key])

    #### Get text embeddings and compare to transcriptome embeddings ####
    logging.info("Computing text embeddings and logits...")
    logits_per_text = score_text_list_vs_transcriptome_emb(
        model,
        text_tokenizer,
        device,
        chunk_size_text_emb_and_scoring,
        score_norm_method,
        transcriptome_embeds,
        text,
    )

    # split text into the different libraries and obs columns, and rank by normalized logits
    logits_df = pd.DataFrame(logits_per_text.numpy(), index=text, columns=["logits"])
    logits_df["term_without_prefix"] = np.nan
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


def score_text_list_vs_transcriptome_emb(
    model,
    text_tokenizer,
    device,
    chunk_size_text_emb_and_scoring,
    score_norm_method,
    transcriptome_embeds,
    text,
):
    """
    Compute the similarity between the text and the transcriptome embeddings.
    :param model: TranscriptomeTextDualEncoderModel instance. Both transcriptome and text embeddings will be computed using this model.
    :param text_tokenizer: AutoTokenizer instance. Used to tokenize the text.
    :param device: torch.device instance
    :param chunk_size_text_emb_and_scoring: int. The text will be chunked into chunks of this size before computing the text \
            embeddings and similarity to the transcriptome. This is necessary to avoid out-of-memory errors.
    :param score_norm_method: "zscore", "softmax", or "01norm". TODO - unclear what is best. How to normalize the logits \
            (similarity to the transcriptome). "zscore" will zscore the logits across all terms. "softmax" will apply softmax to the logits.\
            "01norm" will normalize the logits to the range [0,1].
    :param transcriptome_embeds: torch.tensor. Transcriptome embeddings to compare the text to.
    :param text: List[str]. Text to compare to the transcriptome embeddings.
    :return: torch.tensor. Logits (similarity to the transcriptome) for each term.
    """
    logit_scale = model.logit_scale.exp()

    # Chunk the text to avoid out-of-memory errors
    logits_per_text_list = []
    text_chunks = [
        text[i : i + chunk_size_text_emb_and_scoring]
        for i in range(0, len(text), chunk_size_text_emb_and_scoring)
    ]
    for chunk in text_chunks:
        # Tokenize the chunk and move it to the device
        text_tokens = text_tokenizer(chunk, return_tensors="pt", padding=True)
        for k, v in text_tokens.items():
            text_tokens[k] = v.to(device)

        # Compute text embeddings
        # TODO: It is slow to do this every time. Could store the text embeddings for the enrichR terms and only compute the text embeddings for the obs columns.
        _, text_embeds = model.get_text_features(**text_tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Compute logits (similarity to expression embedding) for the current chunk and append to the list
        logits_per_text = (
            torch.matmul(text_embeds, transcriptome_embeds.t()) * logit_scale
        )
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
