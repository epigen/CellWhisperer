import torch
from torch import nn
import torch.nn.functional as F
from .jsd_info_max import JSDInfoMaxLoss
from typing import Optional


class JSDInfoMaxLossCellWhisperer(JSDInfoMaxLoss):
    """
    JSDInfoMaxLoss is defined by the CLIP-Lite paper
    Arxiv paper for CLIP-Lite: https://arxiv.org/abs/2112.07133
    Repo where JSDInfoMaxLoss is defined: https://github.com/4m4n5/CLIP-Lite/blob/a6825d0258f3876104002fdd9328b8eda0a18746/loss.py#L110C7-L110C21
    """

    def __init__(self, *args, **kwargs):
        super(JSDInfoMaxLossCellWhisperer, self).__init__(*args, **kwargs)

    def forward(
        self,
        transcriptome_features=None,
        text_features=None,
        neg_transcriptome_embeds=None,
        neg_text_embeds=None,
        aug_transcriptome_embeds=None,
        aug_text_embeds=None,
        **kwargs,
    ):
        """
        Args:
        """

        ### This outputs a dictionary with keys {'total_loss','cross_modal_loss','visual_loss','textual_loss'}
        loss_results = super().forward(
            image_features=transcriptome_features,
            text_features=text_features,
            neg_image_features=neg_transcriptome_embeds,
            neg_text_features=neg_text_embeds,
            aug_image_features=aug_transcriptome_embeds,
            aug_text_features=aug_text_embeds,
        )

        return loss_results["total_loss"]


def contrastive_loss(
    logits: torch.Tensor, weight: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device), weight=weight
    )


class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits_per_text: torch.Tensor,
        transcriptome_weights: Optional[torch.Tensor] = None,
        annotation_weights: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the CLIP loss given a similarity matrix between two modalities, typically transcriptome and text embeddings.

        This function calculates the contrastive loss for both the text and transcriptome embeddings using the provided similarity matrix. The similarity matrix is expected to contain cosine similarity scores between pairs of embeddings from the two modalities. The function then averages the losses from both modalities to compute the final CLIP loss.

        The CLIP loss is a symmetric function and encourages the model to align the embeddings from different modalities closer in the embedding space, thereby improving the association between the modalities.

        Args:
            logits_per_text (torch.Tensor): A 2D tensor representing the cosine similarity scores between pairs of embeddings wrt to the text embeddings. The tensor is expected to be square with dimensions [N, N], where N is the number of embeddings in each modality.
            **kwargs: Additional keyword arguments (not currently used).

        Returns:
            torch.Tensor: A scalar tensor representing the computed CLIP loss.

        Note:
            The function internally calls `ContrastiveLoss` twice, once with the similarity matrix (logits_per_text) as is, and once with its transpose (logits_per_transcriptome), to calculate the losses for both modalities (text and transcriptome).
        """
        text_loss = contrastive_loss(logits_per_text, weight=annotation_weights)
        transcriptome_loss = contrastive_loss(
            logits_per_text.t(), weight=transcriptome_weights
        )
        return (text_loss + transcriptome_loss) / 2.0


### The code below is unused and requires refactoring before use
def EuclideanDistance(x, y):
    """
    Compute the Euclidean distance between two sets of embeddings.
    """
    # Compute pairwise distance matrix
    dists = torch.cdist(x, y)
    return dists


def ClipDistanceLoss(
    *, transcriptome_embeds=None, text_embeds=None, margin=1.0, **kwargs
):
    """
    Computes a distance-based loss from a pairwise distance matrix.
    """
    distance_matrix = EuclideanDistance(transcriptome_embeds, text_embeds)

    # Extract diagonal of the matrix, i.e., distances for positive pairs
    positive_distances = torch.diag(distance_matrix)

    # Compute losses for each positive pair
    losses = [
        torch.mean(F.relu(positive_distances[i] - distance_matrix[i] + margin))
        for i in range(len(positive_distances))
    ]

    return torch.mean(torch.stack(losses))


def NormalizeEmbeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Normalize embeddings to have unit L2 norm.
    """
    l2_norms = embeddings.norm(p=2, dim=1, keepdim=True)
    return embeddings / l2_norms.clamp(min=1e-12)


def ComputeMMD(
    transcriptome_embeds=None,
    text_embeds=None,
    kernel_type="rbf",
    kernel_mul=2.0,
    kernel_num=5,
    **kwargs,
):
    """
    Compute Maximum Mean Discrepancy between x and y.
    """
    x = transcriptome_embeds
    y = text_embeds
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    # Kernel computations
    if kernel_type == "linear":
        return (
            torch.mean(x.mm(x.t()))
            + torch.mean(y.mm(y.t()))
            - 2 * torch.mean(x.mm(y.t()))
        )
    elif kernel_type == "rbf":
        # RBF kernel calculations
        x_sum = torch.sum(x**2, 1)
        y_sum = torch.sum(y**2, 1)
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = x_sum.unsqueeze(0).expand_as(xx)
        ry = y_sum.unsqueeze(0).expand_as(yy)
        K = torch.exp(-(rx.t() + rx - 2 * xx) / dim * kernel_mul)
        L = torch.exp(-(ry.t() + ry - 2 * yy) / dim * kernel_mul)
        P = torch.exp(-(rx.t() + ry - 2 * zz) / dim * kernel_mul)
        beta = 1.0 / (x_size * (x_size - 1))
        gamma = 2.0 / (x_size * y_size)
        delta = 1.0 / (y_size * (y_size - 1))
        return beta * (torch.sum(K) + torch.sum(L)) - gamma * torch.sum(P)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def CentroidLoss(transcriptome_embeds=None, text_embeds=None, **kwargs):
    transcriptome_mean = torch.mean(transcriptome_embeds, dim=1)
    text_mean = torch.mean(text_embeds, dim=1)
    return torch.norm(transcriptome_mean - text_mean, 2)
