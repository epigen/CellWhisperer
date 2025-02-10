from pathlib import Path
from typing import Tuple
from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor
from cellwhisperer.jointemb.cellwhisperer_lightning import (
    TranscriptomeTextDualEncoderLightning,
)
from cellwhisperer.config import model_path_from_name
from transformers import AutoTokenizer
import torch


def load_cellwhisperer_model(
    model_path: str = None,
    eval: bool = True,
    cache: bool = False,
    transcriptome_model_type: str = None,
) -> Tuple[
    TranscriptomeTextDualEncoderLightning,
    AutoTokenizer,
    TranscriptomeTextDualEncoderProcessor,
]:
    """
    Load a CellWhisperer model from a given path.
    Args:
        model_path: Path to the model. Can be None if transcriptome_model_type is specified.
        eval: Whether to set the model to eval mode.
        cache: Convert both models into frozencached models to enable caching
        transcriptome_model_type: Type of the transcriptome model. Must be one of "geneformer", "scgpt", "uce" or None. If None, model_path must be specified.
    Returns:
        pl_model: The loaded TranscriptomeTextDualEncoderLightning model.
        tokenizer: The tokenizer used for the model.
        transcriptome_processor: The transcriptome processor used for the model.
    """

    assert not (
        model_path is None and transcriptome_model_type is None
    ), "Either model_path or transcriptome_model_type must be specified."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is not None:
        model_path = Path(model_path).expanduser()
        pl_model = TranscriptomeTextDualEncoderLightning.load_from_checkpoint(
            model_path
        )
    else:
        pl_model = TranscriptomeTextDualEncoderLightning(
            model_config={"transcriptome_model_type": transcriptome_model_type},
            loss_config={},
        )
        pl_model.load_pretrained_models(
            transcriptome_model_directory=model_path_from_name(
                transcriptome_model_type
            ),
            text_model_name_or_path=model_path_from_name(
                pl_model.model.text_model.config.model_type
            ),
        )

    # this is for freezing.
    pl_model.freeze()

    if cache:
        # # This is just for allow caching based on `FrozenCachedModels`, you can omit it
        pl_model.model.freeze_models(force_freeze=True)

    if eval:
        pl_model.eval().to(device)
    else:
        pl_model.to(device)

    processor = TranscriptomeTextDualEncoderProcessor(
        pl_model.model.transcriptome_model.config.model_type,
        model_path_from_name(pl_model.model.text_model.config.model_type),
    )

    tokenizer = processor.tokenizer
    transcriptome_processor = processor.transcriptome_processor

    return pl_model, tokenizer, transcriptome_processor
