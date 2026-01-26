# coding: utf-8
import pyarrow
from lightning import Trainer
from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.jointemb.cellwhisperer_lightning import (
    TranscriptomeTextDualEncoderLightning,
    TranscriptomeTextDualEncoderConfig,
)

# config = TranscriptomeTextDualEncoderConfig(
#     projection_dim=512,
#     transcriptome_config={},
#     identity_projection=False,
# )
# config
# model = TranscriptomeTextDualEncoderLightning(model_config=config, loss_config={})
# model.load_pretrained_models(
#     transcriptome_model_directory="../../resources/geneformer-12L-30M",
#     text_model_name_or_path=None,
#     image_model_name_or_path=None,
# )

pl_model, tokenizer, transcriptome_processor, image_processor = (
    load_cellwhisperer_model(
        transcriptome_model_type="geneformer",
    )
)

trainer = Trainer()
try:
    trainer.fit(pl_model)
except:
    pass
trainer.save_checkpoint("spotwhisperer_random.ckpt")
