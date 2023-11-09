from pathlib import Path
import torch

from single_cellm.validation.zero_shot.validate_singlecell_zeroshot import SingleCellZeroshotValidationScoreCalculator
from single_cellm.jointemb.model import TranscriptomeTextDualEncoderModel

# TODO remove hardcoding
geneformer_biogpt_model_path = Path("~/projects/single-cellm/results/models/geneformer-biogpt").expanduser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TranscriptomeTextDualEncoderModel.from_pretrained(geneformer_biogpt_model_path).to(device)
score_calculator = SingleCellZeroshotValidationScoreCalculator()
scores=score_calculator.get_scores(model=model)
print(scores)