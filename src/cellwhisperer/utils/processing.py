import anndata
import torch
from transformers import AutoTokenizer
from typing import Union, List
from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel
from cellwhisperer.jointemb.scgpt_model import ScGPTTranscriptomeProcessor
from cellwhisperer.jointemb.geneformer_model import GeneformerTranscriptomeProcessor


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

    transcriptome_embeds = []
    for i in range(
        0,
        next(iter(transcriptome_processor_result.values())).shape[0],
        batch_size,
    ):
        batch = {
            k: v[i : i + batch_size].to(model.device)
            for k, v in transcriptome_processor_result.items()
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
) -> torch.Tensor:
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
