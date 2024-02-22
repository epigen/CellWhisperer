import numpy as np
import scanpy as sc
import anndata
from cellwhisperer.utils.processing import adata_to_embeds
from scipy import sparse
import scvi
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
    # TODO: Prepare for the case when the transcriptome is too large to fit on the GPU
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


def process_scgpt(adata: anndata.AnnData, models_and_processors_dict: dict) -> None:
    """Return an adata object with scgpt embeddings added as adata.obsm["X_scgpt"]."""
    # features: Output of the transcriptome model, before the projection layer

    adata.obsm[f"X_scgpt"] = (
        adata_to_transcriptome_features(
            adata,
            model=models_and_processors_dict["scgpt"][0],
            transcriptome_processor=models_and_processors_dict["scgpt"][1],
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


def process_hvg(adata: anndata.AnnData, with_PCA=False) -> None:
    """Return an adata object with HVG-based embeddings added as adata.obsm["X_hvg_with_PCA"] or adata.obsm["X_hvg_without_PCA"].
    Procedure: normalize and log1p, then select highly_variable_genes with n_top_genes=2000, then PCA with n_comps=50 (if with_PCA=True).
    """
    adata_for_hvg = adata.copy()
    sc.pp.calculate_qc_metrics(adata_for_hvg, inplace=True)
    sc.pp.normalize_total(adata_for_hvg, target_sum=1e4)
    sc.pp.log1p(adata_for_hvg)
    sc.pp.highly_variable_genes(
        adata_for_hvg, n_top_genes=2000, subset=True, flavor="seurat"
    )  # same flavour as in the zero shot paper
    if with_PCA:  # what I would usually do
        sc.pp.pca(adata_for_hvg)
        adata.obsm[f"X_hvg_with_PCA"] = adata_for_hvg.obsm["X_pca"]
    else:  # As in the zero shot paper notebook
        if sparse.issparse(adata_for_hvg.X):
            adata.obsm[f"X_hvg_without_PCA"] = np.asarray(adata_for_hvg.X.todense())
        else:
            adata.obsm[f"X_hvg_without_PCA"] = adata_for_hvg.X
    del adata_for_hvg
    return adata


def process_all_genes(adata: anndata.AnnData) -> None:
    """
    Return an adata object with all genes embeddings added as adata.obsm["X_all_genes"].
    Procedure: normalize and log1p, then PCA with n_comps=50.
    """
    adata_for_all_genes = adata.copy()
    sc.pp.calculate_qc_metrics(adata_for_all_genes, inplace=True)
    sc.pp.normalize_total(adata_for_all_genes, target_sum=1e4)
    sc.pp.log1p(adata_for_all_genes)
    sc.pp.pca(adata_for_all_genes)
    adata.obsm[f"X_all_genes"] = adata_for_all_genes.obsm["X_pca"]
    del adata_for_all_genes
    return adata


def process_scvi(adata: anndata.AnnData) -> None:
    """Return an adata object with scvi embeddings added as adata.obsm["X_scvi"]."""
    # As in the zero shot paper notebook
    adata_for_scvi = adata.copy()
    adata_for_scvi.layers["counts"] = adata_for_scvi.X
    scvi.model.SCVI.setup_anndata(adata_for_scvi, layer="counts", batch_key="batch")
    model_scvi = scvi.model.SCVI(
        adata_for_scvi, n_layers=2, n_latent=30, gene_likelihood="nb"
    )
    model_scvi.train()
    adata.obsm[f"X_scvi"] = model_scvi.get_latent_representation()
    del adata_for_scvi
    del model_scvi
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
    elif analysis_type == "scgpt":
        adata = process_scgpt(adata, models_and_processors_dict)
    elif analysis_type == "hvg_with_PCA":
        adata = process_hvg(adata, with_PCA=True)
    elif analysis_type == "hvg_without_PCA":
        adata = process_hvg(adata, with_PCA=False)
    elif analysis_type == "all_genes":
        adata = process_all_genes(adata)
    elif analysis_type == "scvi":
        adata = process_scvi(adata)
    else:
        raise ValueError(f"Unsupported analysis type: {analysis_type}")
    return adata
