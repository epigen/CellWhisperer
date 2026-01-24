# coding: utf-8
from lightning import Trainer
from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.jointemb.cellwhisperer_lightning import (
    TranscriptomeTextDualEncoderLightning,
)

config = TranscriptomeTextDualEncoderConfig(
    projection_dim=512,
    transcriptome_model_type="geneformer",
    transcriptome_config={},
    text_model_type="conch_text",
    text_config={"model_type": "conch_text"},
    image_model_type="conch_image",
    image_config={
        "model_type": "conch_image",
        "model_cfg": "conch_ViT-B-16",
        "checkpoint_path": "hf_hub:MahmoodLab/conch",
    },
    identity_projection=False,
)
config
model = TranscriptomeTextDualEncoderLightning(model_config=config, loss_config={})
model.load_pretrained_models(
    transcriptome_model_directory="../..//resources/geneformer-12L-30M",
    text_model_name_or_path=None,
    image_model_name_or_path=None,
)
trainer = Trainer()
trainer.fit(model)
trainer.save_checkpoint(models_path / "spotwhisperer_random.ckpt")
