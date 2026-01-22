import os
import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from conch.open_clip_custom import create_model_from_pretrained


class ConchImageConfig(PretrainedConfig):
    model_type = "conch_image"

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


class ConchImageModel(nn.Module):
    def __init__(self, config: ConchImageConfig):
        super().__init__()
        self.config = config
        token = os.getenv("HUGGINGFACE_TOKEN")
        # return_transform=False, we feed tensors directly
        self.model = create_model_from_pretrained(
            config.model_cfg,
            checkpoint_path=config.checkpoint_path,
            device="cpu",
            return_transform=False,
            hf_auth_token=token,
        )

    def forward(self, patches_ctx=None, patches_cell=None, return_dict: bool = False):
        # Use CONCH's contrastive head (attentional pooler + proj) to preserve alignment
        image_embeds = self.model.encode_image(patches_ctx, proj_contrast=True, normalize=True)
        return (None, image_embeds)

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
            if pretrained_model_name_or_path.startswith("hf_hub:") or \
               "/MahmoodLab/conch" in pretrained_model_name_or_path or \
               pretrained_model_name_or_path == "MahmoodLab/conch":
                cfg = ConchImageConfig(model_cfg="conch_ViT-B-16", checkpoint_path=pretrained_model_name_or_path)
            else:
                ckpt = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
                cfg = ConchImageConfig(model_cfg="conch_ViT-B-16", checkpoint_path=ckpt)
        elif isinstance(cfg, dict):
            cfg = ConchImageConfig(**cfg)
        return cls(cfg)
