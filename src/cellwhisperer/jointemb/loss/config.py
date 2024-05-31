from dataclasses import asdict, dataclass

from cellwhisperer.jointemb.loss.losses import (
    ClipLoss,
    JSDInfoMaxLossCellWhisperer,
)
from torch import nn


@dataclass
class LossConfig:
    """
    Configuration class for loss functions in TranscriptomeTextDualEncoderLightning module.

    A more elegant implementation would 'inherit' individual configs for each of the losses, thus staying modular.
    """

    clip_lambda: float = 1.0
    clip_lite_lambda: float = 0.0
    clip_lite_type: str = "dot"  # NOTE we don't support others at the moment
    clip_lite_prior_weight: float = 0.1
    clip_lite_transcriptome_prior: bool = False  # NOTE would require adding `image_dim`
    clip_lite_text_prior: bool = False
    sample_weighting: bool = True

    def configure_losses(self, discriminator):
        """
        Configures loss functions based on provided configuration.
        """
        loss_functions = [
            {
                "name": "clip",
                "fn": ClipLoss(),
                "lambda": float(self.clip_lambda),
            },
            {
                "name": "clip_lite",
                "fn": JSDInfoMaxLossCellWhisperer(
                    discriminator=discriminator,
                    prior_weight=self.clip_lite_prior_weight,
                    image_prior=self.clip_lite_transcriptome_prior,
                    text_prior=self.clip_lite_text_prior,
                ),
                "lambda": float(self.clip_lite_lambda),
            },
        ]

        return loss_functions

    def to_dict(self):
        return asdict(self)
