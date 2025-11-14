import numpy as np
from scipy import sparse
from tqdm.auto import tqdm
import anndata
import torch
import logging
from cellwhisperer.config import model_path_from_name
from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor
from transformers import AutoTokenizer
from typing import Union, List
from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel


def adata_to_embeds(
    adata: anndata.AnnData,
    model: TranscriptomeTextDualEncoderModel,
    batch_size: int = 32,
    use_image_data: bool = False,
) -> torch.Tensor:
    """
    NOTE: this should become part of model API (like `embed_texts`)

    Compute embeddings for each cell in the adata object - either transcriptome or image based on parameters.
    :param adata: anndata.AnnData instance.
    :param model: TranscriptomeTextDualEncoderModel instance. Used to compute the embeddings.
    :param batch_size: Batch size for processing.
    :param use_image_data: Whether to use image data instead of transcriptome data.
    :return: torch.tensor of embeddings. Shape: n_samples_in_adata * embedding_size (e.g. 512)
    """

    processor = TranscriptomeTextDualEncoderProcessor(
        model.transcriptome_model.config.model_type,
        model_path_from_name(model.text_model.config.model_type),
        model.image_model.config.model_type,
    )

    # Check for image data availability
    has_image_patches = "patches" in adata.obsm
    has_whole_slide_image = "he_slide" in adata.uns
    has_image_model = hasattr(model, "image_model") and model.image_model is not None

    if has_image_patches:
        raise NotImplementedError(
            "Pre-computed image patches not implemented right now."
        )

    if use_image_data and has_image_model:
        if has_whole_slide_image:
            logging.info("Extracting patches from whole slide image using UNIProcessor")
            from cellwhisperer.jointemb.uni_model import UNIProcessor

            uni_processor = UNIProcessor()
            image_embeds = []

            if "wsi_images" in adata.uns and "sample_id" in adata.obs:
                logging.info("Processing multi-sample dataset with multiple WSI images")
                sample_ids = adata.obs["sample_id"].unique()

                for sample_id in sample_ids:
                    if sample_id in adata.uns["wsi_images"]:
                        sample_mask = adata.obs["sample_id"] == sample_id
                        sample_adata = adata[sample_mask].copy()
                        sample_adata.uns["he_slide"] = adata.uns["wsi_images"][
                            sample_id
                        ]

                        patches = uni_processor(sample_adata, return_tensors="pt")
                        batch_inputs = {
                            k: v.to(model.device) for k, v in patches.items()
                        }
                        _, image_embeds_batch = model.get_image_features(
                            **batch_inputs, normalize_embeds=True
                        )
                        image_embeds.append(image_embeds_batch.detach().cpu())

                if image_embeds:
                    return torch.cat(image_embeds, dim=0)
                else:
                    logging.warning(
                        "No patches extracted from multi-sample dataset; falling back to transcriptome"
                    )
            else:
                patches = uni_processor(adata, return_tensors="pt")
                batch_inputs = {k: v.to(model.device) for k, v in patches.items()}
                _, image_embeds = model.get_image_features(
                    **batch_inputs, normalize_embeds=True
                )
                return image_embeds
        else:
            logging.warning(
                "Image data requested but no whole slide image found, falling back to transcriptome"
            )
    elif use_image_data and not has_image_model:
        logging.warning(
            "Image data requested but model has no image capabilities, falling back to transcriptome"
        )

    # Fall back to or use transcriptome embeddings
    logging.info("Using transcriptome embeddings")

    transcriptome_processor_result = processor.transcriptome_processor(
        adata, return_tensors="pt", padding=True
    )

    transcriptome_embeds = []
    n_transcriptomes = next(iter(transcriptome_processor_result.values())).shape[0]
    for i in tqdm(
        range(
            0,
            n_transcriptomes,
            batch_size,
        ),
        desc="Processing transcriptomes",
        total=(n_transcriptomes + batch_size - 1) // batch_size,
        disable=n_transcriptomes < 2 * batch_size,
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
    # Check if the values in the X layer are counts (i.e., integers)
    comp = np.abs(adata.X[:100] - adata.X[:100].astype(int))
    if isinstance(adata.X, sparse.csr_matrix):
        comp = comp.toarray()

    if not np.all(comp < 1e-6):
        try:
            adata.X = adata.layers["counts"]
        except KeyError:
            logging.error(
                "adata.X contains non-integer (probably normalized) counts, but raw counts are not provided in adata.layers['counts']."
            )
