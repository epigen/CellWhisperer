## copied from transformers/models/vision_text_dual_encoder/configuration_vision_text_dual_encoder.py

import logging
from .geneformer_model import GeneformerConfig
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
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and transcriptome projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    ```"""

    model_type = "transcriptome-text-dual-encoder"
    is_composition = True

    def __init__(
        self,
        projection_dim: int = 512,
        logit_scale_init_value: float = 2.6592,
        transcriptome_config: Optional[dict] = None,
        freeze_transcriptome_model: bool = True,
        text_config: Optional[dict] = None,
        freeze_text_model: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if transcriptome_config is None:
            raise ValueError("`transcriptome_config` can not be `None`.")

        if text_config is None:
            raise ValueError("`text_config` can not be `None`.")

        transcriptome_model_type = transcriptome_config.pop("model_type")
        text_model_type = text_config.pop("model_type")

        if transcriptome_model_type == "geneformer":
            self.transcriptome_config = GeneformerConfig(**transcriptome_config)
        else:
            raise ValueError("Unsupported transcriptome model type.")
            self.transcriptome_config = AutoConfig.for_model(
                transcriptome_model_type, **transcriptome_config
            )
        self.freeze_transcriptome_model = freeze_transcriptome_model

        self.text_config = AutoConfig.for_model(text_model_type, **text_config)
        self.freeze_text_model = freeze_text_model

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value

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

        return cls(
            transcriptome_config=transcriptome_config.to_dict(),
            text_config=text_config.to_dict(),
            **kwargs,
        )
