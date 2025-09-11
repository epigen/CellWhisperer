from cellwhisperer.utils.processing import adata_to_embeds
from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel

from typing import Union, List, Optional, Tuple, Dict
import anndata
import logging
import torch
from transformers import AutoTokenizer
from cellwhisperer.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from cellwhisperer.jointemb.scgpt_model import ScGPTTranscriptomeProcessor
from cellwhisperer.jointemb.uce_model import UCETranscriptomeProcessor
from scipy import stats
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np


def score_left_vs_right(
    left_input: Union[anndata.AnnData, torch.Tensor, List[str]],
    right_input: Union[anndata.AnnData, torch.Tensor, List[str]],
    logit_scale: Union[float, torch.Tensor],
    model: Optional[TranscriptomeTextDualEncoderModel] = None,
    average_mode: Optional[str] = "embeddings",
    grouping_keys: Optional[List[str]] = None,
    batch_size: int = 128,
    score_norm_method: Optional[str] = None,
    use_image_data: bool = False,
) -> Tuple[torch.Tensor, Optional[List[str]]]:
    """
    Generic function to compute similarity between left and right inputs via flexible input types.
    Supports any combination of anndata, tensor, and list[str] for both left and right inputs.

    :param left_input: anndata.AnnData, torch.tensor, or List[str]
        If anndata.AnnData, compute embeddings using appropriate modality. If torch.tensor, use provided embeddings. If List[str], compute text embeddings.
    :param right_input: anndata.AnnData, torch.tensor, or List[str]
        If anndata.AnnData, compute embeddings using appropriate modality. If torch.tensor, use provided embeddings. If List[str], compute text embeddings.
    :param logit_scale: float. Scale the logits (similarity) by this factor. Learned CLIP parameter.
    :param model: TranscriptomeTextDualEncoderModel instance. Used to compute embeddings and similarity.
    :param average_mode: "embeddings" or None.
        If "embeddings", first embed each sample, then average the embeddings. If None, don't average, report results at the single-sample level.
    :param grouping_keys: A list with group indicators (one for each sample in left_input). If this is None, while average_mode is not None, all samples are treated as a single group.
    :param batch_size: int. Model processing in batches (to avoid OOM)
    :param score_norm_method: "zscore", "softmax", "01norm" or None. How to normalize the logits
            (similarity scores). "zscore" will zscore the logits across all terms. "softmax" will apply softmax to the logits.
            "01norm" will normalize the logits to the range [0,1]. If None, don't normalize.
    :param use_image_data: bool. If True, and if an input is anndata.AnnData, use image data instead of transcriptome data.
    : return: A tuple of two objects:
        First, a torch.tensor of similarity between the right and left inputs with shape: n_right * n_left.
        Second, the list of left annotations corresponding to the dim1 of the tensors (or None if grouping_keys is None).
    """

    # check inputs
    if type(left_input) == torch.Tensor:
        assert (
            left_input.dim() == 2
        ), f"left_input must be a tensor of shape n_samples * embedding_size, but is {left_input.shape}"
        assert average_mode != "cells", "average_mode='cells' requires adata input"
    if grouping_keys is None and average_mode is not None:
        if isinstance(left_input, torch.Tensor):
            grouping_keys = ["all"] * left_input.shape[0]
        elif isinstance(left_input, list):
            grouping_keys = ["all"] * len(left_input)
        else:  # anndata
            grouping_keys = ["all"] * left_input.shape[0]

    #### Prepare left embeddings ###
    if type(left_input) == torch.Tensor:
        left_embeds = left_input
    elif isinstance(left_input, list):
        # Handle text input for left side
        assert model is not None, "If left input is text, we need the model"
        left_embeds = model.embed_texts(left_input, chunk_size=batch_size)
    else:
        # Handle anndata input for left side
        if average_mode == "cells":
            sorted_unique_annotations = sorted(list(set(grouping_keys)))
            averaged_left_inputs = []
            for annotation in sorted_unique_annotations:
                avg_input_this_value = (
                    left_input[[annotation == x for x in grouping_keys]]
                    .X.mean(axis=0)
                    .A1
                )
                averaged_left_inputs.append(avg_input_this_value)

            averaged_left_inputs = np.stack(averaged_left_inputs)
            grouping_keys = sorted_unique_annotations

            # rebuild adata
            left_input = anndata.AnnData(
                X=averaged_left_inputs,
                obs=pd.DataFrame(index=grouping_keys),
                var=left_input.var,
            )

        left_embeds = adata_to_embeds(
            left_input,
            model,
            batch_size=batch_size,
            use_image_data=use_image_data,
        )

    if average_mode == "embeddings":
        sorted_unique_annotations = sorted(list(set(grouping_keys)))
        averaged_left_embeds = []
        for annotation in sorted_unique_annotations:
            avg_emb_this_group = left_embeds[
                [annotation == x for x in grouping_keys]
            ].mean(dim=0, keepdim=False)
            averaged_left_embeds.append(avg_emb_this_group)
        left_embeds = torch.stack(averaged_left_embeds)
        grouping_keys = sorted_unique_annotations

    #### Prepare right embeddings ###
    if type(right_input) == torch.Tensor:
        right_embeds = right_input
    elif isinstance(right_input, list):
        # Handle text input for right side
        assert model is not None, "If right input is text, we need the model"
        right_embeds = model.embed_texts(right_input, chunk_size=batch_size)
    else:
        # Handle anndata input for right side
        right_embeds = adata_to_embeds(
            right_input,
            model,
            batch_size=batch_size,
            use_image_data=use_image_data,
        )

    if isinstance(logit_scale, torch.Tensor):
        logging.debug("Converting logit_scale to float.")
        logit_scale = logit_scale.item()

    logits_per_right = (
        torch.matmul(right_embeds.cpu(), left_embeds.t().cpu()) * logit_scale
    ).detach()  # n_right * n_left

    # Check for normalization
    assert (
        (torch.norm(right_embeds, dim=1, keepdim=True) - 1) < 1e-3
    ).all(), "Right embeddings are not normalized"
    assert (
        (torch.norm(left_embeds, dim=1, keepdim=True) - 1) < 1e-3
    ).all(), "Left embeddings are not normalized"

    if score_norm_method == "softmax":
        logits_per_right = torch.softmax(logits_per_right, dim=0)
    elif score_norm_method == "01norm":
        logits_per_right = (logits_per_right - logits_per_right.min()) / (
            logits_per_right.max() - logits_per_right.min()
        )
    elif score_norm_method == "zscore":
        logits_per_right = torch.tensor(
            stats.zscore(logits_per_right.numpy(), axis=0), dtype=torch.float32
        )
    elif score_norm_method is None:
        pass
    else:
        raise ValueError(
            f"score_norm_method must be one of 'softmax', 'zscore', '01norm', or None, but is {score_norm_method}"
        )

    return (
        logits_per_right,  # n_right * n_left
        grouping_keys,  # n_left or n_left_groups
    )


# TODO rename to manage images as well
def score_transcriptomes_vs_texts(
    transcriptome_input: Union[anndata.AnnData, torch.Tensor],
    text_list_or_text_embeds: Union[List[str], torch.Tensor],
    logit_scale: Union[float, torch.Tensor],
    model: Optional[TranscriptomeTextDualEncoderModel] = None,
    average_mode: Optional[str] = "embeddings",
    grouping_keys: Optional[List[str]] = None,
    batch_size: int = 128,
    score_norm_method: Optional[str] = None,
    use_image_data: bool = False,
) -> Tuple[torch.Tensor, Optional[List[str]]]:
    """
    Backward compatibility wrapper for score_left_vs_right.
    """
    return score_left_vs_right(
        left_input=transcriptome_input,
        right_input=text_list_or_text_embeds,
        logit_scale=logit_scale,
        model=model,
        average_mode=average_mode,
        grouping_keys=grouping_keys,
        batch_size=batch_size,
        score_norm_method=score_norm_method,
        use_image_data=use_image_data,
    )


def prepare_terms(
    terms: Union[str, Path, Dict], additional_text_dict: Dict = {}
) -> pd.DataFrame:
    """
    Prepare terms for their use with score_transcriptomes_vs_texts

    :param terms: Either a `Path` or `str` to the json file containing the biological terms to match transcriptomes against (e.g. Enrichr) or a dict containing such terms. Expected format in both cases: (keys: library name, values: list of terms)
    :param additional_text_dict: dict. Additional text to compute the similarity to the transcriptome for. \
        Will be embedded as '<key>: <value>'.\
          E.g. if additional_text_dict={"day_of_induction": ["10","20"]}, the similarity to the transcriptome will be computed for \
          "day_of_induction: 10" and "day_of_induction: 20". 
    """
    ### Prepare text ###
    if isinstance(terms, (str, Path)):
        assert os.path.exists(terms), f"terms json path {terms} does not exist"

        # Load terms (e.g. EnrichR)
        with open(terms, "r") as f:
            terms: Dict = json.load(f)

    # Convert the Dict[str, List[str]] to a DataFrame with columns "library" and "term"
    terms_df = pd.DataFrame(
        [
            (lib, term)
            for lib in {**terms, **additional_text_dict}.keys()
            for term in terms[lib]
        ],
        columns=["library", "term"],
    )
    return terms_df


def rank_terms_by_score(scores: torch.Tensor, terms_df: pd.DataFrame) -> pd.DataFrame:
    """
    modifies the input df
    """
    terms_df["logits"] = scores[:, 0]  # n_text * 1 (so second dim is 'empty'

    terms_df["rank_in_library"] = terms_df.groupby("library")["logits"].rank(
        ascending=False
    )
    terms_df["rank_total"] = terms_df["logits"].rank(ascending=False)

    return terms_df.sort_values(by="logits", ascending=False)


def formatted_text_from_df(df, n_top_per_term):
    """
    Format the output of ranke_terms_by_score() as a string.
    :param df: pd.DataFrame. Output of ranke_terms_by_score().
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
                f"{row['term']} ({row['logits']:.2f})"
                for _, row in top_n_terms.iterrows()
            ]
        )
        top_n_text = f"{library}:\n\t{top_n_text}"
        top_n_per_split.append(top_n_text)

    return "\n".join(top_n_per_split)
