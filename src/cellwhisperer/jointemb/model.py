# from /home/moritz/Projects/transformers/src/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py
""" PyTorch TranscriptomeTextDualEncoder model."""

from typing import Optional, Tuple, Union, Any, List
from cellwhisperer.config import model_path_from_name
from cellwhisperer.jointemb.frozen_model import FrozenCachedModel
from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor
from cellwhisperer.config import get_path


import torch
from .geneformer_model import GeneformerConfig, GeneformerModel
from .scgpt_model import ScGPTConfig, ScGPTModel
from .uce_model import UCEConfig, UCEModel
from .loss.discriminator import GlobalDiscriminatorDot

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    logging,
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
    text_features: BaseModelOutputWithPooling = None
    transcriptome_features: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            (
                self[k]
                if k not in ["text_features", "transcriptome_features"]
                else getattr(self, k).to_tuple()
            )
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

        # initialize with config (makes self.config available)
        super().__init__(config)

        if transcriptome_model is None:
            if isinstance(config.transcriptome_config, GeneformerConfig):
                transcriptome_model = GeneformerModel(config.transcriptome_config)
            elif isinstance(config.transcriptome_config, ScGPTConfig):
                transcriptome_model = ScGPTModel(config.transcriptome_config)
            elif isinstance(config.transcriptome_config, UCEConfig):
                transcriptome_model = UCEModel(config.transcriptome_config)
            else:
                raise NotImplementedError(
                    "Only geneformer, scgpt and uce are supported for now"
                )

        if text_model is None:
            text_model = AutoModel.from_config(self.config.text_config)

        self.prepare_models(transcriptome_model, text_model)
        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.transcriptome_model.config = self.config.transcriptome_config
        self.text_model.config = self.config.text_config

        try:
            self.transcriptome_embed_dim = config.transcriptome_config.hidden_size
        except AttributeError:  # UCE
            self.transcriptome_embed_dim = config.transcriptome_config.output_dim
        self.text_embed_dim = config.text_config.hidden_size
        self.projection_dim = config.projection_dim

        self.discriminator = GlobalDiscriminatorDot(
            image_sz=self.transcriptome_embed_dim,
            text_sz=self.text_embed_dim,
            units=self.projection_dim,
            bln=True,  # batch layer norm
        )

    def prepare_models(self, transcriptome_model, text_model, force_freeze=False):
        """
        Freeze the transcriptome and text model if indicated by self.config

        Comparing to "L" (see below) is important to retain the correct weights in the model for checkpoint loading

        Args:
            transcriptome_model (*): The transcriptome model to be used.
            text_model (PreTrainedModel): The text model to be used.
            force_freeze (bool): Whether to force freezing the models even if the config does not indicate it.
        """
        if self.config.locking_mode[0] == "L" or force_freeze:
            if not isinstance(transcriptome_model, FrozenCachedModel):
                transcriptome_model = FrozenCachedModel(transcriptome_model)
        elif self.config.unlocked_fp16:
            transcriptome_model.half()

        assert (
            text_model is not None
        ), "text_model must be provided"  # doesn't make sense that only transcriptome_model gets initialized before

        if self.config.locking_mode[1] == "L" or force_freeze:
            if not isinstance(text_model, FrozenCachedModel):
                text_model = FrozenCachedModel(text_model)
        elif self.config.unlocked_fp16:
            text_model.half()

        self.text_model = text_model
        self.transcriptome_model = transcriptome_model

    def freeze_models(self, force_freeze=False):
        """
        Freeze the transcriptome and text model (if they are not marked as "u"). They will get unfrozen after an warmup phase (if marked as "L")

        Args:
            transcriptome_model (*): The transcriptome model to be used.
            text_model (PreTrainedModel): The text model to be used.
            force_freeze (bool): Whether to force freezing the models even if the config does not indicate it.
        """
        device = self.device

        if self.config.locking_mode[0] != "u" or force_freeze:
            if not isinstance(self.transcriptome_model, FrozenCachedModel):
                self.transcriptome_model = FrozenCachedModel(self.transcriptome_model)

        if self.config.locking_mode[1] != "u" or force_freeze:
            if not isinstance(self.text_model, FrozenCachedModel):
                self.text_model = FrozenCachedModel(self.text_model)

        # need to restore .device parameter (it is to cpu() by this operation)
        self.to(device)

    def unfreeze_U_towers(self):
        """
        Unfreeze the transcriptome and text model according to config

        .train() is called, as this method is called in the train loop `on_train_epoch_batch_end` in the `LightningModule`). See https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks
        """

        if self.config.locking_mode[0] == "U":
            self.transcriptome_model = self.transcriptome_model.model.train().to(
                self.device
            )
            if self.config.unlocked_fp16:
                self.transcriptome_model.half()
        if self.config.locking_mode[1] == "U":
            self.text_model = self.text_model.model.train().to(self.device)
            if self.config.unlocked_fp16:
                self.text_model.half()

    def _text_pooling(self, text_outputs: Tuple, attention_mask: torch.FloatTensor):
        if isinstance(text_outputs[1], torch.Tensor):
            assert not torch.isnan(text_outputs[1]).any()
            return text_outputs[
                1
            ].float()  # counter implicit conversion into bfloat16 in dense layer in BERT pooler
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
        expression_gene=None,
        expression_expr=None,
        expression_key_padding_mask=None,
        normalize_embeds=False,
        return_dict=None,
    ):
        r"""

        Returns:
            transcriptome_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The transcriptome embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTranscriptomeModel`].

        """

        transcriptome_features = self.transcriptome_model(
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            expression_gene=expression_gene,
            expression_expr=expression_expr,
            expression_key_padding_mask=expression_key_padding_mask,
            return_dict=False,
        )[1]

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
        expression_gene: Optional[torch.LongTensor] = None,
        expression_expr: Optional[torch.LongTensor] = None,
        expression_key_padding_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], CLIPOutput]:
        assert attention_mask is not None, "Attention mask must be provided"
        assert (
            expression_tokens is not None and expression_token_lengths is not None
        ) or (
            expression_expr is not None
            # and expression_gene is not None  # not needed for UCE
            and expression_key_padding_mask is not None
        ), "Either expression_tokens and expression_token_lengths or expression_gene, expression_expr and expression_key_padding_mask must be provided"

        transcriptome_features = self.transcriptome_model(
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            expression_gene=expression_gene,
            expression_expr=expression_expr,
            expression_key_padding_mask=expression_key_padding_mask,
            return_dict=False,
        )[1]

        text_features = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
            **kwargs,
        )
        text_features = self._text_pooling(text_features, attention_mask)

        (
            logits_per_transcriptome,
            transcriptome_embeds,
            text_embeds,
        ) = self.discriminator(transcriptome_features, text_features)
        logits_per_text = logits_per_transcriptome.T

        if not return_dict:
            return (
                logits_per_transcriptome,
                logits_per_text,
                text_embeds,
                transcriptome_embeds,
                text_features,
                transcriptome_features,
            )

        return CLIPOutput(
            logits_per_transcriptome=logits_per_transcriptome,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            transcriptome_embeds=transcriptome_embeds,
            text_features=text_features,
            transcriptome_features=transcriptome_features,
        )

    def store_cache(self):
        """
        Save cached transcriptome and text embeddings/features, if the corresponding models have been frozen.
        """
        try:
            self.transcriptome_model.save_cache()
        except AttributeError:
            pass

        try:
            self.text_model.save_cache()
        except AttributeError:
            pass

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
            elif kwargs_transcriptome["config"]["model_type"] == "scgpt":
                kwargs_transcriptome["config"] = ScGPTConfig(
                    **kwargs_transcriptome["config"]
                )
                transcriptome_model = ScGPTModel.from_pretrained(
                    f"{transcriptome_model_name_or_path}/best_model.pt",
                    # *model_args,  # these args are not supported by geneformer
                    **kwargs_transcriptome,
                )
            elif kwargs_transcriptome["config"]["model_type"] == "uce":
                kwargs_transcriptome["config"] = UCEConfig(
                    **kwargs_transcriptome["config"]
                )

                transcriptome_model = UCEModel.from_pretrained(
                    transcriptome_model_name_or_path,
                    get_path(["uce_paths", "tokens"]),
                    config=kwargs_transcriptome["config"],
                )

            else:
                raise NotImplementedError("Only geneformer and scgpt are supported")
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

        logger.debug(
            "The projection layer and logit scale weights `['transcriptome_projection.weight', 'text_projection.weight',"
            " 'logit_scale']` are freshly initialized."
        )

        return model

    # inference API
    def embed_texts(self, texts: List[str], chunk_size=64):
        """
        Embed the given texts into the LLM space
        """

        processor = TranscriptomeTextDualEncoderProcessor(
            self.transcriptome_model.config.model_type,
            model_path_from_name(self.text_model.config.model_type),
        )

        tokenizer = processor.tokenizer

        ret_list = []
        for chunk in [
            texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)
        ]:
            text_tokens = tokenizer(chunk, return_tensors="pt", padding=True)
            for k, v in text_tokens.items():
                text_tokens[k] = v.to(self.device)

            # Compute text embeddings
            _, text_embeds = self.get_text_features(**text_tokens)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            ret_list.append(text_embeds)

        return torch.cat(ret_list, dim=0)
