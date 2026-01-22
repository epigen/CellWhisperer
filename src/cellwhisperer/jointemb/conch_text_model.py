import os
import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from conch.open_clip_custom import create_model_from_pretrained


class ConchTextConfig(PretrainedConfig):
    model_type = "conch_text"

    def __init__(
        self,
        model_cfg: str = "conch_ViT-B-16",
        checkpoint_path: str = "hf_hub:MahmoodLab/conch",
        embed_dim: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        self.embed_dim = int(embed_dim)
        # Align with model expectations: text hidden size used by pipeline
        self.hidden_size = int(embed_dim)


class ConchTextModel(nn.Module):
    def __init__(self, config: ConchTextConfig):
        super().__init__()
        self.config = config
        token = os.getenv("HUGGINGFACE_TOKEN")
        # Use same underlying model object; we'll only call encode_text

        self.model = create_model_from_pretrained(
            config.model_cfg,
            checkpoint_path=config.checkpoint_path,
            device="cpu",
            return_transform=False,
            hf_auth_token=token,
        )

    def forward(self, input_ids=None, attention_mask=None, return_dict: bool = False):
        # input_ids is expected to be pre-tokenized with CONCH tokenizer
        text_embeds = self.model.encode_text(input_ids)
        # Return in a shape compatible with _text_pooling (will take index 1)
        return (None, text_embeds)

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        cfg = kwargs.pop("config", None)
        if cfg is None:
            if (
                pretrained_model_name_or_path.startswith("hf_hub:")
                or "/MahmoodLab/conch" in pretrained_model_name_or_path
                or pretrained_model_name_or_path == "MahmoodLab/conch"
            ):
                cfg = ConchTextConfig(
                    model_cfg="conch_ViT-B-16",
                    checkpoint_path=pretrained_model_name_or_path,
                )
            else:
                ckpt = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
                cfg = ConchTextConfig(model_cfg="conch_ViT-B-16", checkpoint_path=ckpt)
        elif isinstance(cfg, dict):
            cfg = ConchTextConfig(**cfg)
        return cls(cfg)
