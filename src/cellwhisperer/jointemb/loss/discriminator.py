"""
These lines of codes are defined in source code provided with the CLIP-Lite paper
Arxiv paper for CLIP-Lite: https://arxiv.org/abs/2112.07133
Repo https://github.com/4m4n5/CLIP-Lite

Note: The original source code has been slightly modified by us
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np


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
