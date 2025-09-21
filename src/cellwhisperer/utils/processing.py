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
from cellwhisperer.jointemb.scgpt_model import ScGPTTranscriptomeProcessor
from cellwhisperer.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from cellwhisperer.jointemb.uce_model import UCETranscriptomeProcessor


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
    has_whole_slide_image = "20x_slide" in adata.uns
    has_image_model = hasattr(model, "image_model") and model.image_model is not None

    if use_image_data and has_image_model:
        if has_image_patches:
            logging.info("Using pre-computed image patches")
            # Extract image patches and process them
            image_patches = adata.obsm["patches"]
            image_embeds = []

            # Process patches in batches
            num_patches = len(image_patches)
            for i in tqdm(
                range(0, num_patches, batch_size),
                desc="Processing image patches",
                total=(num_patches + batch_size - 1) // batch_size,
                disable=num_patches < 2 * batch_size,
            ):
                batch_patches = image_patches[i : i + batch_size]

                # Process the batch of patches
                processed_batch = processor.image_processor(
                    images=batch_patches, return_tensors="pt"
                )

                # Move to device and get embeddings
                batch_inputs = {
                    k: v.to(model.device) for k, v in processed_batch.items()
                }
                _, image_embeds_batch = model.get_image_features(
                    **batch_inputs, normalize_embeds=True
                )
                image_embeds.append(image_embeds_batch.detach().cpu())

            image_embeds = torch.cat(image_embeds, dim=0)
            return image_embeds

        elif has_whole_slide_image:
            logging.info("Extracting patches from whole slide image using UniProcessor")
            # Import here to avoid circular imports
            from cellwhisperer.jointemb.uni_model import UNIProcessor

            # Extract patches using UniProcessor
            uni_processor = UNIProcessor()

            # For combined datasets with multiple WSI images, we need to process each sample separately
            if "wsi_images" in adata.uns and "sample_id" in adata.obs:
                # Multi-sample dataset - process each sample with its corresponding WSI
                logging.info("Processing multi-sample dataset with multiple WSI images")
                all_patches = []
                sample_ids = adata.obs["sample_id"].unique()

                for sample_id in sample_ids:
                    if sample_id in adata.uns["wsi_images"]:
                        sample_mask = adata.obs["sample_id"] == sample_id
                        sample_adata = adata[sample_mask].copy()

                        # Set the correct WSI for this sample
                        sample_adata.uns["20x_slide"] = adata.uns["wsi_images"][
                            sample_id
                        ]

                        # Extract patches for this sample
                        patches_data = uni_processor(sample_adata, return_tensors="pt")
                        all_patches.append(patches_data["patches"])

                        logging.info(
                            f"Extracted {len(patches_data['patches'])} patches for sample {sample_id}"
                        )

                # Combine patches from all samples
                if all_patches:
                    combined_patches = torch.cat(all_patches, dim=0)
                    adata.obsm["patches"] = combined_patches
                else:
                    logging.warning("No patches extracted from multi-sample dataset")
            else:
                # Single WSI dataset - use default processing
                patches_data = uni_processor(adata, return_tensors="pt")
                adata.obsm["patches"] = patches_data["patches"]

            # Now process the extracted patches (recursive call with patches now available)
            return adata_to_embeds(adata, model, batch_size, use_image_data=True)
        else:
            logging.warning(
                "Image data requested but no patches or whole slide image found, falling back to transcriptome"
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
