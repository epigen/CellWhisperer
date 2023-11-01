import torch
from torch import nn


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
