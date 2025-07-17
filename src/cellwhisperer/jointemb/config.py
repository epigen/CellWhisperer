## copied from transformers/models/vision_text_dual_encoder/configuration_vision_text_dual_encoder.py

import logging
from .geneformer_model import GeneformerConfig
from cellwhisperer.config import model_path_from_name
from .scgpt_model import ScGPTConfig
from .uce_model import UCEConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto.configuration_auto import AutoConfig
from typing import Any, Dict, Optional


logger = logging.get_logger(__name__)


class TranscriptomeTextDualEncoderConfig(PretrainedConfig):
    r"""
    See documentation in VisionTextDualEncoderConfig

    [`TranscriptomeTextDualEncoderConfig`] is the configuration class to store the configuration of a
    [`TranscriptomeTextDualEncoderModel`]. It is used to instantiate [`TranscriptomeTextDualEncoderModel`] model according to the
    specified arguments, defining the text model and transcriptome model configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        projection_dim (`int`, *optional*, defaults to 1024):
            Dimentionality of text and transcriptome projection layers.
        transcriptome_config (`Dict`, *optional*):
            Dictionary of transcriptome model configuration parameters.
        text_config (`Dict`, *optional*):
            Dictionary of text model configuration parameters.
        locking_mode (`str`, defaults to "LU"): Follows 'LiT' paper convention. The first letter corresponds to the training mode for the transcriptome model, the second to the text model. 'L' for locked, 'U' for unfrozen, 'u' for unfrozen and randomly initialized
        unlocked_fp16 (`bool`, defaults to False): Whether to use fp16 for the unlocked models.

        kwargs (*optional*):
            Dictionary of keyword arguments.

    ```"""

    model_type = "transcriptome-text-dual-encoder"
    is_composition = True

    def __init__(
        self,
        projection_dim: int = 1024,
        transcriptome_model_type: str = "geneformer",
        transcriptome_config: Dict = {},
        text_model_type: str = "bert",
        text_config: Dict = {},
        locking_mode: str = "LU",
        unlocked_fp16: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if transcriptome_config is None:
            raise ValueError("`transcriptome_config` can not be `None`.")

        if text_config is None:
            raise ValueError("`text_config` can not be `None`.")

        transcriptome_config = transcriptome_config.copy()
        text_config = text_config.copy()

        self.locking_mode = locking_mode
        self.unlocked_fp16 = unlocked_fp16

        if transcriptome_model_type == "geneformer":
            self.transcriptome_config = GeneformerConfig(**transcriptome_config)
        elif transcriptome_model_type == "scgpt":
            self.transcriptome_config = ScGPTConfig(**transcriptome_config)
        elif transcriptome_model_type == "uce":
            self.transcriptome_config = UCEConfig(**transcriptome_config)
        else:
            raise ValueError(
                f"Unsupported transcriptome model type: {transcriptome_model_type}"
            )
            self.transcriptome_config = AutoConfig.from_pretrained(
                model_path_from_name(transcriptome_model_type), **transcriptome_config
            )

        self.text_config = AutoConfig.from_pretrained(
            model_path_from_name(text_model_type), **text_config
        )

        self.projection_dim = int(
            projection_dim
        )  # workaround Lightning CLI not interpreting the string as int as expected

    @classmethod
    def from_transcriptome_text_configs(
        cls,
        transcriptome_config: PretrainedConfig,
        text_config: PretrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`TranscriptomeTextDualEncoderConfig`] (or a derived class) from text model configuration and transcriptome
        model configuration.

        Returns:
            [`TranscriptomeTextDualEncoderConfig`]: An instance of a configuration object
        """

        transcriptome_config = transcriptome_config.to_dict()
        transcriptome_model_type = transcriptome_config.pop("model_type")

        text_config = text_config.to_dict()
        text_model_type = text_config.pop("model_type")

        return cls(
            transcriptome_model_type=transcriptome_model_type,
            transcriptome_config=transcriptome_config,
            text_model_type=text_model_type,
            text_config=text_config,
            **kwargs,
        )
