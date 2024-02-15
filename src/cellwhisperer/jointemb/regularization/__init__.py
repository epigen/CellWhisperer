"""
Regularization methods for the model.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InputRegularization(nn.Module):
    """
    Regularization methods for the model.
    """

    def __init__(
        self,
        gauss_noise_std: Optional[float] = 0.0,
    ):
        """
        Args:
            gauss_noise_std: Standard deviation of the Gaussian noise to add to the input embeddings during training.
        """
        super().__init__()
        self.gauss_noise_std = gauss_noise_std

    def forward(self, outputs):
        # Conditionally add noise to embeddings during training

        logging.warning(
            "InputRegularization is being used. This needs to be tested properly first"
        )

        if (
            self.training
            and self.gauss_noise_std is not None
            and self.gauss_noise_std > 0.0
        ):
            with torch.no_grad():  # Ensure no gradient is computed for the noise addition
                if outputs.text_embeds is not None:
                    noise = torch.randn_like(outputs.text_embeds) * self.gauss_noise_std
                    outputs.text_embeds = outputs.text_embeds + noise
                if outputs.transcriptome_embeds is not None:
                    noise = (
                        torch.randn_like(outputs.transcriptome_embeds)
                        * self.gauss_noise_std
                    )
                    outputs.transcriptome_embeds = outputs.transcriptome_embeds + noise

        return outputs
