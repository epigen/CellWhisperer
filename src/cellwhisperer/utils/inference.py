from cellwhisperer.utils.processing import adata_to_embeds, text_list_to_embeds
from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel

from typing import Union, List, Optional, Tuple
import anndata
import torch
from transformers import AutoTokenizer
from cellwhisperer.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from cellwhisperer.jointemb.scgpt_model import ScGPTTranscriptomeProcessor
from scipy import stats


def score_transcriptomes_vs_texts(
    model: TranscriptomeTextDualEncoderModel,
    transcriptome_input: Union[anndata.AnnData, torch.Tensor],
    text_list_or_text_embeds: Union[List[str], torch.Tensor],
    average_mode: Optional[str] = "embeddings",
    grouping_keys: Optional[List[str]] = None,
    text_tokenizer: Optional[AutoTokenizer] = None,
    transcriptome_processor: Optional[
        Union[GeneformerTranscriptomeProcessor, ScGPTTranscriptomeProcessor]
    ] = None,
    batch_size: int = 128,
    score_norm_method: Optional[str] = "zscore",
) -> Tuple[torch.Tensor, Optional[List[str]]]:
    """
    Convenience function to compute the similarity between text and transcriptome (embeddings) via flexible inputs

    :param model: TranscriptomeTextDualEncoderModel instance. Used to compute the similarity between text and transcriptome.
    :param transcriptome_input: anndata.AnnData or torch.tensor (n_cells*embedding_size) \
        If anndata.AnnData, first compute the transcriptome embeddings. If torch.tensor, use the provided transcriptome embeddings.
    :param text_list_or_text_embeds: List[str] or torch.tensor (n_celltype * embedding_size). If List[str], compute the text embeddings for each text. \
        If torch.tensor, use the provided text embeddings.
    :param average_mode: "cells" or "embeddings" or None. If "cells", first average the transcriptome data across all cells of same celltype, then tokenize and embed. \
        If "embeddings", first tokenize and embed each cell, then average the embeddings. If None, don't average, report scores at the single-transcriptome level. TODO "cells" does not work yet.
    :param grouping_keys: A list with group indicators (one for each transcriptome in transcriptome_input). If average_mode is not None, this must be provided and will be used to split the transciptome_input into groups that will be averaged separately.
    :param text_tokenizer: AutoTokenizer instance. Used to tokenize the text. Can be None if text_list_or_text_embeds is a torch.tensor.
    :param transcriptome_processor: GeneformerTranscriptomeProcessor or ScGPTTranscriptomeProcessor instance. Used to prepare/tokenize the transcriptome. Can be None if transcriptome_input is a torch.tensor.
    :param batch_size: int. Model processing in batches (to avoid OOM)
    :param score_norm_method: "zscore", "softmax", "01norm" or None. TODO - unclear what is best. How to normalize the logits \
            (similarity to the transcriptome). "zscore" will zscore the logits across all terms. "softmax" will apply softmax to the logits.\
            "01norm" will normalize the logits to the range [0,1]. If None, don't normalize.
    : return: A tuple of two objects: \
        First, a torch.tensor of  similarity between the text and the adatas with shape: n_text * n_adata. \
        Second, the list of transcriptome annotations corresponding to the dim1 of the tensors (or None if grouping_keys is None).
    """
    logit_scale = model.discriminator.temperature.exp()

    # check inputs
    if type(transcriptome_input) == torch.Tensor:
        assert (
            transcriptome_input.dim() == 2
        ), f"transcriptome_input must be a tensor of shape n_cells * embedding_size, but is {transcriptome_input.shape}"
    assert not (
        grouping_keys is None and average_mode is not None
    ), "grouping_keys must be provided if average_mode is not None"

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
        grouping_keys,  # n_cells or n_celltypes
    )
