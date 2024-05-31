"""
JSDInfoMaxLoss is defined by the CLIP-Lite paper
Arxiv paper for CLIP-Lite: https://arxiv.org/abs/2112.07133
Repo where JSDInfoMaxLoss is defined: https://github.com/4m4n5/CLIP-Lite/blob/a6825d0258f3876104002fdd9328b8eda0a18746/loss.py#L110C7-L110C21

Note: The original source code has been slightly modified by us
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum
import math
import numpy as np

# from .utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MILinearBlock(nn.Module):
    def __init__(self, feature_sz, units=2048, bln=True):
        super(MILinearBlock, self).__init__()
        # Pre-dot product encoder for "Encode and Dot" arch for 1D feature maps
        self.feature_nonlinear = nn.Sequential(
            nn.Linear(feature_sz, units, bias=False),
            nn.BatchNorm1d(units),
            nn.ReLU(),
            nn.Linear(units, units),
        )
        self.feature_shortcut = nn.Linear(feature_sz, units)
        self.feature_block_ln = nn.LayerNorm(units)

        # initialize the initial projection to a sort of noisy copy
        eye_mask = np.zeros(
            (units, feature_sz), dtype=np.bool_
        )  ## seems like np.bool is deprecated
        for i in range(min(feature_sz, units)):
            eye_mask[i, i] = 1

        self.feature_shortcut.weight.data.uniform_(-0.01, 0.01)
        self.feature_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.0)
        self.bln = bln

    def forward(self, feat):
        f = self.feature_nonlinear(feat) + self.feature_shortcut(feat)
        if self.bln:
            f = self.feature_block_ln(f)

        return f


class PriorDiscriminator(nn.Module):
    def __init__(self, sz):
        super(PriorDiscriminator, self).__init__()
        self.l0 = nn.Linear(sz, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class GlobalDiscriminator(nn.Module):
    def __init__(self, sz):
        super(GlobalDiscriminator, self).__init__()
        self.l0 = nn.Linear(sz, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, features1, features2):
        x = torch.cat((features1, features2), dim=1)
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))

        return self.l2(h)


@torch.jit.script
def norm_and_dot(x, y, temp):
    return torch.dot(x, y) * temp


class GlobalDiscriminatorDot(nn.Module):
    def __init__(self, image_sz, text_sz, units=2048, bln=True):
        super(GlobalDiscriminatorDot, self).__init__()
        self.img_block = MILinearBlock(image_sz, units=units, bln=bln)
        self.text_block = MILinearBlock(text_sz, units=units, bln=bln)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        features1=None,
        features2=None,
    ):
        """
        Compute full matrix dot product (each vs each)
        """
        # Computer cross modal loss
        feat1 = self.img_block(features1)
        feat2 = self.text_block(features2)

        feat1, feat2 = map(lambda t: F.normalize(t, p=2, dim=-1), (feat1, feat2))

        # ## Method 1
        # # Dot product and sum
        # o = torch.mm(feat1, feat2.t()) * self.temperature.exp()

        # ## Method 2
        # o = self.cos(feat1.unsqueeze(1), feat2.unsqueeze(0)) * self.temperature.exp()

        # Method 3
        o = torch.einsum("nd,md->nm", [feat1, feat2]) * self.temperature.exp()

        return o, feat1, feat2


class JSDInfoMaxLoss(nn.Module):
    def __init__(
        self,
        discriminator,
        image_dim=2048,
        text_dim=768,
        prior_weight=0.1,
        image_prior=True,
        text_prior=False,
        visual_self_supervised=False,
        textual_self_supervised=False,
    ):
        super().__init__()

        # Settings to be saved for forward
        self.prior_weight = prior_weight
        self.image_prior = image_prior
        self.text_prior = text_prior

        type = "dot"  # NOTE: we don't support others at the moment
        self.global_d = discriminator
        if type == "concat":
            if visual_self_supervised:
                self.visual_d = GlobalDiscriminator(sz=image_dim + image_dim)
            if textual_self_supervised:
                self.textual_d = GlobalDiscriminator(sz=text_dim + text_dim)

        elif type == "dot":
            if visual_self_supervised:
                self.visual_d = GlobalDiscriminatorDot(
                    image_sz=image_dim, text_sz=image_dim
                )
            if textual_self_supervised:
                self.textual_d = GlobalDiscriminatorDot(
                    image_sz=text_dim, text_sz=text_dim
                )

        elif type == "condot":
            if visual_self_supervised:
                self.visual_d = GlobalDiscriminatorDot(
                    image_sz=image_dim, text_sz=image_dim
                )
            if textual_self_supervised:
                self.textual_d = GlobalDiscriminatorDot(
                    image_sz=text_dim, text_sz=text_dim
                )

        elif type == "dotcon":
            if visual_self_supervised:
                self.visual_d = GlobalDiscriminator(sz=image_dim + image_dim)
            if textual_self_supervised:
                self.textual_d = GlobalDiscriminator(sz=text_dim + text_dim)

        if self.image_prior:
            self.prior_d = PriorDiscriminator(sz=image_dim)
        if self.text_prior:
            self.text_prior_d = PriorDiscriminator(sz=text_dim)

    def forward(
        self,
        image_features,
        text_features,
        neg_image_features=None,
        neg_text_features=None,
        aug_image_features=None,
        aug_text_features=None,
    ):
        # Prior losses
        PRIOR = torch.tensor(0.0).to(device)  # changed from .cuda()
        # Image prior loss
        if self.image_prior:
            image_prior = torch.rand_like(image_features)
            term_a = torch.log(self.prior_d(image_prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(image_features)).mean()
            IMAGE_PRIOR = -(term_a + term_b)
            PRIOR = PRIOR + IMAGE_PRIOR
        # Text prior loss
        if self.text_prior:
            text_prior = torch.rand_like(text_features)
            term_a = torch.log(self.text_prior_d(text_prior)).mean()
            term_b = torch.log(1.0 - self.text_prior_d(text_features)).mean()
            TEXT_PRIOR = -(term_a + term_b)
            PRIOR = PRIOR + TEXT_PRIOR

        # Cross modal MI maximization loss
        # Normal mode
        if neg_text_features is None:
            # Positive pairs
            Ej = -F.softplus(
                -self.global_d(  # NOTE here we could optimize by passing the result of this computation
                    features1=image_features,
                    features2=text_features,
                )[
                    0
                ].diag()
            ).mean()

            # Negative pairs
            text_features_prime = torch.cat(
                (text_features[1:], text_features[0].unsqueeze(0)), dim=0
            )
            Em = F.softplus(
                self.global_d(  # NOTE if we compute negative pairs within our model.py, we could get rid of global_d in this loss class altogether
                    features1=image_features,
                    features2=text_features_prime,
                )[
                    0
                ].diag()
            ).mean()

        # Cluster mode
        elif neg_text_features is not None:
            # Positive pairs
            image_features_all = torch.cat((image_features, neg_image_features), dim=0)
            text_features_all = torch.cat((text_features, neg_text_features), dim=0)
            Ej = -F.softplus(
                -self.global_d(  # NOTE here we could optimize by passing the result of this computation
                    features1=image_features_all,
                    features2=text_features_all,
                )[
                    0
                ].diag()
            ).mean()

            # Shuffle text_features so have half batch does not have hard negatives
            text_features = torch.cat(
                (text_features[1:], text_features[0].unsqueeze(0)), dim=0
            )

            # Negative pairs
            text_features_prime_all = torch.cat(
                (neg_text_features, text_features), dim=0
            )
            Em = F.softplus(
                self.global_d(
                    features1=image_features_all,
                    features2=text_features_prime_all,
                )[0].diag()
            ).mean()

        CROSS_MODAL_LOSS = Em - Ej

        # Visual self supervised loss
        VISUAL_LOSS = torch.tensor(0.0).to(device)  # changed from .cuda()
        if aug_image_features is not None:
            # Positive pairs
            Ej = -F.softplus(
                -self.visual_d(
                    features1=image_features,
                    features2=aug_image_features,
                )
            ).mean()
            # Negative pairs
            aug_image_features_prime = torch.cat(
                (aug_image_features[1:], aug_image_features[0].unsqueeze(0)), dim=0
            )
            Em = F.softplus(
                self.visual_d(
                    features1=image_features,
                    features2=aug_image_features_prime,
                )
            ).mean()

            VISUAL_LOSS = Em - Ej

        # Textal self supervised loss
        TEXTUAL_LOSS = torch.tensor(0.0).to(device)  # changed from .cuda()
        if aug_text_features is not None:
            # Positive pairs
            Ej = -F.softplus(
                -self.textual_d(
                    features1=text_features,
                    features2=aug_text_features,
                )
            ).mean()
            # Negative pairs
            aug_text_features_prime = torch.cat(
                (aug_text_features[1:], aug_text_features[0].unsqueeze(0)), dim=0
            )
            Em = F.softplus(
                self.textual_d(
                    features1=text_features,
                    features2=aug_text_features_prime,
                )
            ).mean()

            TEXTUAL_LOSS = Em - Ej

        JSD_LOSS = CROSS_MODAL_LOSS + VISUAL_LOSS + TEXTUAL_LOSS
        TOTAL_LOSS = ((1.0 - self.prior_weight) * JSD_LOSS) + (
            self.prior_weight * PRIOR
        )

        loss_dict = {
            "total_loss": TOTAL_LOSS,
            "cross_modal_loss": CROSS_MODAL_LOSS,
            "visual_loss": VISUAL_LOSS,
            "textual_loss": TEXTUAL_LOSS,
        }

        return loss_dict
