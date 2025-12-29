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

    has_whole_slide_image = (
        "he_slide" in adata.uns or "20x_slide" in adata.uns or "image_path" in adata.uns
    )  # 20x_slide is legacy for HEST, image_path for file-based images
    has_image_model = hasattr(model, "image_model") and model.image_model is not None

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
    """
    Ensure that adata.X contains raw (integer) counts.
    If not, try to switch to adata.layers["counts"].
    We test on the first 100 cells to keep this cheap.
    """

    def _is_raw_counts(mat, n_cells: int = 1000) -> bool:
        # Take a small sample of cells
        sample = mat[:n_cells]
        # Convert to dense if sparse
        if sparse.issparse(sample):
            sample = sample.toarray()
        else:
            sample = np.asarray(sample)
        comp = np.abs(sample - sample.astype(int))
        if np.all(comp < 1e-6):
            return True
        else:
            # check if there are values > 30
            if np.any(sample > 30):
                return True
            else:
                return False

    # First, check adata.X
    if not _is_raw_counts(adata.X):
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

        if not _is_raw_counts(counts):
            logging.error(
                "adata.layers['counts'] also does not appear to contain raw integer counts."
            )
            raise ValueError(
                "Neither adata.X nor adata.layers['counts'] look like raw integer counts."
            )

        # If we get here, counts looks good → use it as X
        adata.X = counts
