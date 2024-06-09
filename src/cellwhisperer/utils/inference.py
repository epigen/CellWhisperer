from cellwhisperer.utils.processing import adata_to_embeds
from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel

from typing import Union, List, Optional, Tuple, Dict
import anndata
import logging
import torch
from transformers import AutoTokenizer
from cellwhisperer.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from cellwhisperer.jointemb.scgpt_model import ScGPTTranscriptomeProcessor
from scipy import stats
import os
import json
from pathlib import Path
import pandas as pd


def score_transcriptomes_vs_texts(
    transcriptome_input: Union[anndata.AnnData, torch.Tensor],
    text_list_or_text_embeds: Union[List[str], torch.Tensor],
    logit_scale: Union[float, torch.Tensor],
    model: Optional[TranscriptomeTextDualEncoderModel] = None,
    average_mode: Optional[str] = "embeddings",
    grouping_keys: Optional[List[str]] = None,
    transcriptome_processor: Optional[
        Union[GeneformerTranscriptomeProcessor, ScGPTTranscriptomeProcessor]
    ] = None,
    batch_size: int = 128,
    score_norm_method: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[List[str]]]:
    """
    Convenience function to compute the similarity between text and transcriptome (embeddings) via flexible inputs

    :param transcriptome_input: anndata.AnnData or torch.tensor (n_cells*embedding_size) \
        If anndata.AnnData, first compute the transcriptome embeddings. If torch.tensor, use the provided transcriptome embeddings.
    :param text_list_or_text_embeds: List[str] or torch.tensor (n_celltype * embedding_size). If List[str], compute the text embeddings for each text. \
        If torch.tensor, use the provided text embeddings.
    :param logit_scale: float. Scale the logits (similarity to the transcriptome) by this factor. Learned CLIP parameter.
    :param model: TranscriptomeTextDualEncoderModel instance. Used to compute the similarity between text and transcriptome.
    :param average_mode: "embeddings" or None. \
        If "embeddings", first tokenize and embed each cell, then average the embeddings. If None, don't average, report results at the single-transcriptome level. NOTE "cells" is not implemented at the moment but would work by first averaging the transcriptome data across all cells of same celltype, then tokenize and embed. \
    :param grouping_keys: A list with group indicators (one for each transcriptome in transcriptome_input). If this is None, while average_mode is not None, all transcriptomes are treated as a single group.
    :param transcriptome_processor: GeneformerTranscriptomeProcessor or ScGPTTranscriptomeProcessor instance. Used to prepare/tokenize the transcriptome. Can be None if transcriptome_input is a torch.tensor.
    :param batch_size: int. Model processing in batches (to avoid OOM)
    :param score_norm_method: "zscore", "softmax", "01norm" or None. How to normalize the logits \
            (similarity to the transcriptome). "zscore" will zscore the logits across all terms. "softmax" will apply softmax to the logits.\
            "01norm" will normalize the logits to the range [0,1]. If None, don't normalize.
    : return: A tuple of two objects: \
        First, a torch.tensor of  similarity between the text and the adatas with shape: n_text * n_adata. \
        Second, the list of transcriptome annotations corresponding to the dim1 of the tensors (or None if grouping_keys is None).
    """

    # check inputs
    if type(transcriptome_input) == torch.Tensor:
        assert (
            transcriptome_input.dim() == 2
        ), f"transcriptome_input must be a tensor of shape n_cells * embedding_size, but is {transcriptome_input.shape}"
    if grouping_keys is None and average_mode is not None:
        grouping_keys = ["all"] * transcriptome_input.shape[0]

    #### Prepare transcriptome embeddings ###
    if average_mode == "cells":
        raise NotImplementedError("average_mode='cells' not implemented yet")

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
        sorted_unique_annotations = sorted(list(set(grouping_keys)))
        averaged_transcriptome_embeds = []
        for annotation in sorted_unique_annotations:
            avg_emb_this_celltype = transcriptome_embeds[
                [annotation == x for x in grouping_keys]
            ].mean(
                dim=0, keepdim=False
            )  # 512
            averaged_transcriptome_embeds.append(avg_emb_this_celltype)
        transcriptome_embeds = torch.stack(
            averaged_transcriptome_embeds
        )  # n_celltypes * 512
        grouping_keys = sorted_unique_annotations

    if type(text_list_or_text_embeds) == torch.Tensor:
        text_embeds = text_list_or_text_embeds
    else:
        assert (
            model is not None
        ), "If text is provided as string, we need the model (is None)"
        text_embeds = model.embed_texts(text_list_or_text_embeds, chunk_size=batch_size)

    if isinstance(logit_scale, torch.Tensor):
        logging.debug("Converting logit_scale to float.")
        logit_scale = logit_scale.item()

    logits_per_text = (
        torch.matmul(text_embeds.cpu(), transcriptome_embeds.t().cpu()) * logit_scale
    ).detach()  # n_text * n_adatas

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
        grouping_keys,  # n_cells or n_celltypes
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
