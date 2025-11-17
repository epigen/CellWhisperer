import numpy as np
from scipy import sparse
import anndata
import torch
import logging
from transformers import AutoTokenizer
from typing import Union, List
from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel
from cellwhisperer.jointemb.scgpt_model import ScGPTTranscriptomeProcessor
from cellwhisperer.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from cellwhisperer.jointemb.uce_model import UCETranscriptomeProcessor


def adata_to_embeds(
    adata: anndata.AnnData,
    model: TranscriptomeTextDualEncoderModel,
    transcriptome_processor: Union[
        GeneformerTranscriptomeProcessor,
        ScGPTTranscriptomeProcessor,
        UCETranscriptomeProcessor,
    ],
    batch_size: int = 32,
) -> torch.Tensor:
    """
    NOTE: this should become part of model API (like `embed_texts`)

    Compute the transcriptome embeddings for each cell in the adata object.
    :param adata: anndata.AnnData instance.
    :param model: TranscriptomeTextDualEncoderModel instance. Used to compute the transcriptome embeddings.
    :param transcriptome_processor: GeneformerTranscriptomeProcessor, UCETranscriptomeProcessor or ScGPTTranscriptomeProcessor instance. Used to prepare/tokenize the transcriptome.
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


def ensure_raw_counts_adata(adata):
    """
    Ensure that adata.X contains raw (integer) counts.
    If not, try to switch to adata.layers["counts"].
    We test on the first 100 cells to keep this cheap.
    """

    def _is_integer_counts(mat, n_cells: int = 100) -> bool:
        # Take a small sample of cells
        sample = mat[:n_cells]
        # Convert to dense if sparse
        if sparse.issparse(sample):
            sample = sample.toarray()
        else:
            sample = np.asarray(sample)
        comp = np.abs(sample - sample.astype(int))
        return np.all(comp < 1e-6)

    # First, check adata.X
    if not _is_integer_counts(adata.X):
        # Try to fall back to raw counts in layers["counts"]
        try:
            counts = adata.layers["counts"]
        except KeyError:
            logging.error(
                "adata.X contains non-integer (probably normalized) counts, "
                "but raw counts are not provided in adata.layers['counts']."
            )
            raise ValueError(
                "adata.X does not appear to contain raw integer counts, "
                "and no adata.layers['counts'] is available."
            )

        if not _is_integer_counts(counts):
            logging.error(
                "adata.layers['counts'] also does not appear to contain raw integer counts."
            )
            raise ValueError(
                "Neither adata.X nor adata.layers['counts'] look like raw integer counts."
            )

        # If we get here, counts looks good → use it as X
        adata.X = counts
