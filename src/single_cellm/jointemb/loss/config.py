from dataclasses import asdict, dataclass

from single_cellm.jointemb.loss.losses import (
    ClipLoss,
    JSDInfoMaxLossSingleCeLLM,
)
from torch import nn


@dataclass
class LossConfig:
    """
    Configuration class for loss functions in TranscriptomeTextDualEncoderLightning module.

    A more elegant implementation would 'inherit' individual configs for each of the losses, thus staying modular.
    """

    clip_lambda: float = 0.0
    clip_lite_lambda: float = 1.0
    clip_lite_type: str = "dot"  # TODO we don't support others at the moment
    clip_lite_prior_weight: float = 0.1
    clip_lite_transcriptome_prior: bool = (
        False  # TODO requires image_dim (no big deal to add)
    )
    clip_lite_text_prior: bool = False

    def configure_losses(self, projection_dim, discriminator):
        """
        Configures loss functions based on provided configuration.
        """
        loss_functions = nn.ModuleDict(
            {
                "clip": ClipLoss(self.clip_lambda),
                "clip_lite": JSDInfoMaxLossSingleCeLLM(
                    weight=self.clip_lite_lambda,
                    discriminator=discriminator,
                    prior_weight=self.clip_lite_prior_weight,
                    image_prior=self.clip_lite_transcriptome_prior,
                    text_prior=self.clip_lite_text_prior,
                ),
            }
        )

        return loss_functions

    def to_dict(self):
        return asdict(self)
