# from /home/moritz/Projects/transformers/src/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py
""" PyTorch TranscriptomeTextDualEncoder model."""


from typing import Optional, Tuple, Union, Any
from single_cellm.config import get_path
from single_cellm.jointemb.frozen_model import FrozenCachedModel

import torch
from pathlib import Path
from torch import nn
from .geneformer_model import GeneformerConfig, GeneformerModel
from clip_lite.loss import GlobalDiscriminatorDot

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    ModelOutput,
)
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel

from dataclasses import dataclass
from .config import TranscriptomeTextDualEncoderConfig

logger = logging.get_logger(__name__)


@dataclass
class CLIPOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits_per_transcriptome: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    transcriptome_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    transcriptome_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["text_model_output", "transcriptome_model_output"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class TranscriptomeTextDualEncoderModel(PreTrainedModel):
    """
    For more docs refer to the original VisionTextDualEncoderModel
    """

    config_class = TranscriptomeTextDualEncoderConfig
    base_model_prefix = "transcriptome_text_dual_encoder"

    def __init__(
        self,
        config: Optional[TranscriptomeTextDualEncoderConfig] = None,
        transcriptome_model: Optional[Union[PreTrainedModel, FrozenCachedModel]] = None,
        text_model: Optional[Union[PreTrainedModel, FrozenCachedModel]] = None,
    ):
        if config is None and (transcriptome_model is None or text_model is None):
            raise ValueError(
                "Either a configuration or an transcriptome and a text model has to be provided"
            )

        if config is None:
            config = TranscriptomeTextDualEncoderConfig.from_transcriptome_text_configs(
                transcriptome_model.config, text_model.config
            )
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(
                    f"config: {config} has to be of type {self.config_class}"
                )

        # initialize with config
        super().__init__(config)

        if transcriptome_model is None:
            if isinstance(config.transcriptome_config, GeneformerConfig):
                transcriptome_model = GeneformerModel(config.transcriptome_config)
            else:
                transcriptome_model = AutoModel.from_config(  # TODO support our transcriptome model natively
                    config.transcriptome_config
                )

        if config.locking_mode[0] == "L":
            if not isinstance(transcriptome_model, FrozenCachedModel):
                transcriptome_model = FrozenCachedModel(
                    transcriptome_model,
                    get_path(["paths", "transcriptome_model_cache"]),
                )
        elif config.unlocked_fp16:
            transcriptome_model.half()

        if text_model is None:
            text_model = AutoModel.from_config(config.text_config)

        if config.locking_mode[1] == "L":
            if not isinstance(text_model, FrozenCachedModel):
                text_model = FrozenCachedModel(
                    text_model, get_path(["paths", "text_model_cache"])
                )
        elif config.unlocked_fp16:
            text_model.half()

        self.text_model = text_model
        self.transcriptome_model = transcriptome_model
        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.transcriptome_model.config = self.config.transcriptome_config
        self.text_model.config = self.config.text_config

        self.transcriptome_embed_dim = config.transcriptome_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.projection_dim = config.projection_dim

        self.discriminator = GlobalDiscriminatorDot(
            image_sz=self.transcriptome_embed_dim,
            text_sz=self.text_embed_dim,
            units=self.projection_dim,
            bln=True,  # batch layer norm
        )

    def _text_pooling(self, text_outputs: Tuple, attention_mask: torch.FloatTensor):
        """
        TODO wouldn't it make sense to scale the embeddings by the attention itself?
        """
        if isinstance(text_outputs[1], torch.Tensor):
            return text_outputs[1]
        token_embeddings = text_outputs[0]

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        return mean_pooled

    def get_text_features(
        self,
        input_ids=None,
        attention_mask=None,
        normalize_embeds=False,
        return_dict=None,
        **kwargs,
    ):
        r"""
        Returns:
           text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].
        ```"""
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            # token_type_ids=token_type_ids,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        text_features = self._text_pooling(text_outputs, attention_mask)

        text_embeds = self.discriminator.text_block(text_features)

        if normalize_embeds:
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return text_features, text_embeds

    def get_transcriptome_features(
        self,
        expression_tokens=None,
        expression_token_lengths=None,
        # output_attentions=None,  # TODO what is this?
        # output_hidden_states=None,
        normalize_embeds=False,
        return_dict=None,
    ):
        r"""

        Returns:
            transcriptome_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The transcriptome embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTranscriptomeModel`].

        ```"""

        transcriptome_features = self.transcriptome_model(
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        transcriptome_embeds = self.discriminator.img_block(transcriptome_features)

        if normalize_embeds:
            transcriptome_embeds = transcriptome_embeds / transcriptome_embeds.norm(
                dim=-1, keepdim=True
            )

        return transcriptome_features, transcriptome_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        expression_tokens: Optional[torch.FloatTensor] = None,
        expression_token_lengths: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # token_type_ids: Optional[torch.LongTensor] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], CLIPOutput]:
        # assert output_attentions is None
        # assert output_hidden_states is None

        transcriptome_outputs = self.transcriptome_model(
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            return_dict=False,
        )[0]

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            # token_type_ids=token_type_ids,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            return_dict=False,
            **kwargs,
        )
        text_outputs = self._text_pooling(text_outputs, attention_mask)

        (
            logits_per_transcriptome,
            transcriptome_embeds,
            text_embeds,
        ) = self.discriminator(transcriptome_outputs, text_outputs)
        logits_per_text = logits_per_transcriptome.T

        if not return_dict:
            return (
                logits_per_transcriptome,
                logits_per_text,
                text_embeds,
                transcriptome_embeds,
                text_outputs,
                transcriptome_outputs,
            )

        return CLIPOutput(
            logits_per_transcriptome=logits_per_transcriptome,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            transcriptome_embeds=transcriptome_embeds,
            text_model_output=text_outputs,  # TODO rename to text_features
            transcriptome_model_output=transcriptome_outputs,  # TODO rename transcriptome_features
        )

    def store_cache(self):
        """
        Save the cached transcriptome and text models to a given path.

        Args:
            path (Path): Path to save the cached models to.
        """
        if self.config.locking_mode[0] == "L":
            self.transcriptome_model.save_cache()

        if self.config.locking_mode[1] == "L":
            self.text_model.save_cache()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_transcriptome_text_pretrained(
        cls,
        transcriptome_model_name_or_path: Optional[str] = None,
        text_model_name_or_path: Optional[str] = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        """
        Documentation below was copied and trimmed from the `transformers` library and still largely applies.

        Params:
            transcriptome_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the transcriptome model. Can be either:
            text_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the text model. Can be either:
            model_args (remaining positional arguments, *optional*):
                All remaning positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).
        ```"""
        kwargs_transcriptome = {
            argument[len("transcriptome_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("transcriptome_")
        }

        kwargs_text = {
            argument[len("text_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("text_")
        }

        # remove transcriptome, text kwargs from kwargs
        for key in kwargs_transcriptome.keys():
            del kwargs["transcriptome_" + key]
        for key in kwargs_text.keys():
            del kwargs["text_" + key]

        # Load and initialize the transcriptome and text model
        transcriptome_model = kwargs_transcriptome.pop("model", None)
        if transcriptome_model is None:
            if transcriptome_model_name_or_path is None:
                raise ValueError(
                    "If `transcriptome_model` is not defined as an argument, a `transcriptome_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_transcriptome:
                transcriptome_config = AutoConfig.from_pretrained(
                    transcriptome_model_name_or_path
                )
                kwargs_transcriptome["config"] = transcriptome_config

            if kwargs_transcriptome["config"]["model_type"] == "geneformer":
                kwargs_transcriptome["config"] = GeneformerConfig(
                    **kwargs_transcriptome["config"]
                )
                transcriptome_model = GeneformerModel.from_pretrained(
                    transcriptome_model_name_or_path,
                    # *model_args,  # these args are not supported by geneformer
                    **kwargs_transcriptome,
                )
            else:
                raise NotImplementedError("Only geneformer is supported")
                kwargs_transcriptome["config"] = transcriptome_config
                transcriptome_model = AutoModel.from_pretrained(
                    transcriptome_model_name_or_path,
                    *model_args,
                    **kwargs_transcriptome,
                )

        text_model = kwargs_text.pop("model", None)
        if text_model is None:
            if text_model_name_or_path is None:
                raise ValueError(
                    "If `text_model` is not defined as an argument, a `text_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_text:
                text_config = AutoConfig.from_pretrained(text_model_name_or_path)
                kwargs_text["config"] = text_config

            text_model = AutoModel.from_pretrained(
                text_model_name_or_path, *model_args, **kwargs_text
            )

        # instantiate config with corresponding kwargs
        config = TranscriptomeTextDualEncoderConfig.from_transcriptome_text_configs(
            transcriptome_model.config, text_model.config, **kwargs
        )

        # init model
        model = cls(
            config=config,
            transcriptome_model=transcriptome_model,
            text_model=text_model,
        )

        # the projection layers are always newly initialized when loading the model
        # using pre-trained transcriptome and text model.
        logger.warning(
            "The projection layer and logit scale weights `['transcriptome_projection.weight', 'text_projection.weight',"
            " 'logit_scale']` are newly initialized. You should probably TRAIN this model on a down-stream task to be"
            " able to use it for predictions and inference."
        )

        return model
