import numpy as np
import scanpy as sc
import anndata
from cellwhisperer.utils.processing import adata_to_embeds
from scipy import sparse
from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel
from cellwhisperer.jointemb.processing import (
    GeneformerTranscriptomeProcessor,
    ScGPTTranscriptomeProcessor,
)
import torch
from typing import Union


def adata_to_transcriptome_features(
    adata: anndata.AnnData,
    model: TranscriptomeTextDualEncoderModel,
    transcriptome_processor: Union[
        GeneformerTranscriptomeProcessor, ScGPTTranscriptomeProcessor
    ],
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Compute the transcriptome features for each cell in the adata object.
    :param adata: anndata.AnnData instance.
    :param model: TranscriptomeTextDualEncoderModel instance. Used to compute the transcriptome embeddings.
    :param transcriptome_processor: GeneformerTranscriptomeProcessor or ScGPTTranscriptomeProcessor instance. Used to prepare/tokenize the transcriptome.
    :return: torch.tensor of transcriptome embeddings. Shape: n_transcriptomes_in_adata * embedding_size (e.g. 512)
    """

    transcriptome_processor_result = transcriptome_processor(
        adata, return_tensors="pt", padding=True
    )
    # make sure transcriptome_tokens are on GPU
    for k, v in transcriptome_processor_result.items():
        transcriptome_processor_result[k] = v.to(model.device)

    transcriptome_features = []
    for i in range(
        0,
        next(iter(transcriptome_processor_result.values())).shape[0],
        batch_size,
    ):
        batch = {
            k: v[i : i + batch_size] for k, v in transcriptome_processor_result.items()
        }
        transcriptome_features_batch, _ = model.get_transcriptome_features(**batch)
        transcriptome_features.append(transcriptome_features_batch)
    transcriptome_features = torch.cat(transcriptome_features, dim=0)

    transcriptome_features = transcriptome_features / transcriptome_features.norm(
        dim=-1, keepdim=True
    )

    return transcriptome_features


def process_cellwhisperer(
    adata: anndata.AnnData, models_and_processors_dict: dict
) -> None:
    """Return an adata object with cellwhisperer embeddings added as adata.obsm["X_cellwhisperer"]."""
    # embeds: The full output of the model, using the projection layer
    adata.obsm[f"X_cellwhisperer"] = (
        adata_to_embeds(
            adata,
            model=models_and_processors_dict["cellwhisperer"][0],
            transcriptome_processor=models_and_processors_dict["cellwhisperer"][1],
        )
        .cpu()
        .numpy()
    )
    return adata

def process_geneformer(
    adata: anndata.AnnData, models_and_processors_dict: dict
) -> None:
    """Return an adata object with geneformer embeddings added as adata.obsm["X_geneformer"]."""
    # features: Output of the transcriptome model, before the projection layer
    adata.obsm[f"X_geneformer"] = (
        adata_to_transcriptome_features(
            adata,
            model=models_and_processors_dict["geneformer"][0],
            transcriptome_processor=models_and_processors_dict["geneformer"][1],
        )
        .cpu()
        .numpy()
    )
    return adata

def get_adata_with_embedding(
    adata: anndata.AnnData, models_and_processors_dict: dict, analysis_type: str
) -> None:
    """Create and add embeddings to the anndata, based on the provided analysis type."""

    adata = adata.copy()

    if analysis_type == "cellwhisperer":
        adata = process_cellwhisperer(adata, models_and_processors_dict)
    elif analysis_type == "geneformer":
        adata = process_geneformer(adata, models_and_processors_dict)
    else:
        raise ValueError(f"Unsupported analysis type: {analysis_type}")
    return adata
