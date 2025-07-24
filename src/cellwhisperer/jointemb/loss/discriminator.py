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
    def __init__(self, transcriptome_sz, text_sz, image_sz, units=2048, bln=True):
        super(GlobalDiscriminatorDot, self).__init__()
        self.transcriptome_block = MILinearBlock(transcriptome_sz, units=units, bln=bln)
        self.text_block = MILinearBlock(text_sz, units=units, bln=bln)
        self.image_block = MILinearBlock(image_sz, units=units, bln=bln)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        transcripome_features=None,
        text_features=None,
        image_features=None,
        text_batch_mask=None,
        image_batch_mask=None,
        transcriptome_batch_mask=None,
    ):
        """
        Compute cross modal loss, i.e. compute full matrix dot product (each vs each)
        
        Handle forward pass with modality masks to compute only valid pairwise comparisons.
        Returns all three possible logits matrices separately, with None for invalid pairs.
        
        Args:
            transcripome_features: Transcriptome feature tensor
            text_features: Text feature tensor  
            image_features: Image feature tensor
            text_batch_mask: Boolean mask indicating which samples have text data
            image_batch_mask: Boolean mask indicating which samples have image data
            transcriptome_batch_mask: Boolean mask indicating which samples have transcriptome data
            
        TODO: could test negatively contrasting against non-matching counterparts
        """

        embeds = []
        for fn, features in zip(
            (self.transcriptome_block, self.text_block, self.image_block),
            [transcripome_features, text_features, image_features],
        ):
            if features is not None:
                embeds.append(fn(features))
            else:
                embeds.append(None)

        transcriptome_embed, text_embed, image_embed = embeds
        
        # Normalize embeddings if they exist
        if transcriptome_embed is not None:
            transcriptome_embed = F.normalize(transcriptome_embed, p=2, dim=-1)
        if text_embed is not None:
            text_embed = F.normalize(text_embed, p=2, dim=-1)
        if image_embed is not None:
            image_embed = F.normalize(image_embed, p=2, dim=-1)
        
        # Initialize outputs as None
        logits_transcriptome_text = None
        logits_transcriptome_image = None
        logits_text_image = None
        
        # Compute transcriptome-text pairs
        if (transcriptome_embed is not None and text_embed is not None and 
            transcriptome_batch_mask is not None and text_batch_mask is not None):
            
            # Get valid pairs (samples that have both modalities)
            valid_pairs = transcriptome_batch_mask & text_batch_mask
            if valid_pairs.any():
                t_embed = transcriptome_embed[valid_pairs]
                txt_embed = text_embed[valid_pairs]
                if len(t_embed) > 0 and len(txt_embed) > 0:
                    logits_transcriptome_text = torch.einsum("nd,md->nm", [t_embed, txt_embed]) * self.temperature.exp()
        
        # Compute transcriptome-image pairs  
        if (transcriptome_embed is not None and image_embed is not None and
            transcriptome_batch_mask is not None and image_batch_mask is not None):
            
            valid_pairs = transcriptome_batch_mask & image_batch_mask
            if valid_pairs.any():
                t_embed = transcriptome_embed[valid_pairs]
                img_embed = image_embed[valid_pairs]
                if len(t_embed) > 0 and len(img_embed) > 0:
                    logits_transcriptome_image = torch.einsum("nd,md->nm", [t_embed, img_embed]) * self.temperature.exp()
        
        # Compute text-image pairs
        if (text_embed is not None and image_embed is not None and
            text_batch_mask is not None and image_batch_mask is not None):
            
            valid_pairs = text_batch_mask & image_batch_mask
            if valid_pairs.any():
                txt_embed = text_embed[valid_pairs]
                img_embed = image_embed[valid_pairs]
                if len(txt_embed) > 0 and len(img_embed) > 0:
                    logits_text_image = torch.einsum("nd,md->nm", [txt_embed, img_embed]) * self.temperature.exp()
        
        # Check if we have at least one valid pair
        if logits_transcriptome_text is None and logits_transcriptome_image is None and logits_text_image is None:
            # No valid pairs found, raise ValueError
            raise ValueError("No valid modality pairs found for loss computation")
        
        # Return all three logits matrices (with None for missing pairs) and embeddings
        return (logits_transcriptome_text, logits_transcriptome_image, logits_text_image), transcriptome_embed, text_embed, image_embed
