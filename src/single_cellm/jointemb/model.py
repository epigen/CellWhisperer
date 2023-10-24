# from /home/moritz/Projects/transformers/src/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py
""" PyTorch TranscriptomeTextDualEncoder model."""


from typing import Optional, Tuple, Union, Any

import torch
from torch import nn
from .geneformer_model import GeneformerConfig, GeneformerModel

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
from .config import TranscriptomeTextDualEncoderConfig  # TODO make relative again

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


# Copied from transformers.models.clip.modeling_clip.contrastive_loss
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


# Copied from transformers.models.clip.modeling_clip.clip_loss
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    transcriptome_loss = contrastive_loss(similarity.t())
    return (caption_loss + transcriptome_loss) / 2.0


class TranscriptomeTextDualEncoderModel(PreTrainedModel):
    """
    For more docs refer to the original VisionTextDualEncoderModel
    """

    config_class = TranscriptomeTextDualEncoderConfig
    base_model_prefix = "transcriptome_text_dual_encoder"

    def __init__(
        self,
        config: Optional[TranscriptomeTextDualEncoderConfig] = None,
        transcriptome_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
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

        if text_model is None:
            text_model = AutoModel.from_config(config.text_config)

        self.transcriptome_model = transcriptome_model
        self.text_model = text_model

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.transcriptome_model.config = self.config.transcriptome_config
        self.text_model.config = self.config.text_config

        self.transcriptome_embed_dim = config.transcriptome_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.projection_dim = config.projection_dim

        self.transcriptome_projection = nn.Linear(
            self.transcriptome_embed_dim, self.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias=False
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(self.config.logit_scale_init_value)
        )

    def get_text_features(
        self,
        input_ids=None,
        attention_mask=None,
        # position_ids=None,
        # token_type_ids=None,
        # output_attentions=None,
        # output_hidden_states=None,
        return_dict=None,
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
        )

        # mean pooling
        token_embeddings = text_outputs[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # text_embeds = text_outputs[1]  # pooler_output, this works for BERT I believe
        text_embeds = self.text_projection(mean_pooled)

        return text_outputs, text_embeds

    def get_transcriptome_features(
        self,
        expression_tokens=None,
        expression_token_lengths=None,
        # output_attentions=None,
        # output_hidden_states=None,
        pad_token_id=None,
        return_dict=None,
    ):
        r"""

        Returns:
            transcriptome_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The transcriptome embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTranscriptomeModel`].

        ```"""
        if pad_token_id is None:
            raise ValueError(
                "pad_token_id has to be defined when using `get_transcriptome_features`"
            )

        transcriptome_outputs = self.transcriptome_model(
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            pad_token_id=pad_token_id,
            return_dict=return_dict,
        )

        # pooled_output = transcriptome_outputs[1]  # pooled_output
        transcriptome_embeds = self.transcriptome_projection(transcriptome_outputs)

        return transcriptome_outputs, transcriptome_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        expression_tokens: Optional[torch.FloatTensor] = None,
        expression_token_lengths: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        # token_type_ids: Optional[torch.LongTensor] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        pad_token_id: Any = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CLIPOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        # assert output_attentions is None
        # assert output_hidden_states is None

        transcriptome_outputs, transcriptome_embeds = self.get_transcriptome_features(
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            pad_token_id=pad_token_id,
        )

        text_outputs, text_embeds = self.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # normalized features
        transcriptome_embeds = transcriptome_embeds / transcriptome_embeds.norm(
            dim=-1, keepdim=True
        )
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = (
            torch.matmul(text_embeds, transcriptome_embeds.t()) * logit_scale
        )
        logits_per_transcriptome = logits_per_text.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (
                logits_per_transcriptome,
                logits_per_text,
                text_embeds,
                transcriptome_embeds,
                text_outputs,
                transcriptome_outputs,
            )
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_transcriptome=logits_per_transcriptome,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            transcriptome_embeds=transcriptome_embeds,
            text_model_output=text_outputs,
            transcriptome_model_output=transcriptome_outputs,
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_transcriptome_text_pretrained(
        cls,
        transcriptome_model_name_or_path: str = None,
        text_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        """
        Params:
            transcriptome_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the transcriptome model. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *PyTorch checkpoint folder* (e.g, `./pt_model`). In this case, `from_pt`
                      should be set to `True` and a configuration object should be provided as `config` argument. This
                      loading path is slower than converting the PyTorch checkpoint in a Flax model using the provided
                      conversion scripts and loading the Flax model afterwards.

            text_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the text model. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *PyTorch checkpoint folder* (e.g, `./pt_model`). In this case, `from_pt`
                      should be set to `True` and a configuration object should be provided as `config` argument. This
                      loading path is slower than converting the PyTorch checkpoint in a Flax model using the provided
                      conversion scripts and loading the Flax model afterwards.

            model_args (remaining positional arguments, *optional*):
                All remaning positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the text configuration, use the prefix *text_* for each configuration parameter.
                - To update the transcriptome configuration, use the prefix *transcriptome_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import TranscriptomeTextDualEncoderModel

        >>> # initialize a model from pretrained ViT and BERT models. Note that the projection layers will be randomly initialized.
        >>> model = TranscriptomeTextDualEncoderModel.from_transcriptome_text_pretrained(
        ...     "google/vit-base-patch16-224", "bert-base-uncased"
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./vit-bert")
        >>> # load fine-tuned model
        >>> model = TranscriptomeTextDualEncoderModel.from_pretrained("./vit-bert")
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

            if transcriptome_config.model_type == "geneformer":
                kwargs_transcriptome[
                    "config"
                ] = transcriptome_config.transcriptome_config
                transcriptome_model = GeneformerModel.from_pretrained(
                    transcriptome_model_name_or_path,
                    *model_args,
                    **kwargs_transcriptome,
                )
                # TODO: Should we use the pre-trained projection as well ?
            else:
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
