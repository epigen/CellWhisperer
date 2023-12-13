from .zero_shot.cancer_gene_essentiality import EvaluateCancerGeneEssentiality
from .zero_shot.single_cell_annotation import (
    SingleCellZeroshotValidationScoreCalculator,
    TOP20_LUNG_LIVER_BLOOD_CELLTYPES,
)


def initialize_validation_functions(
    batch_size: int, transcriptome_model_type: str, text_model_type: str
):
    training_validation_functions = {
        "zero_shot_cancer_gene_essentiality": EvaluateCancerGeneEssentiality(
            batch_size, transcriptome_model_type, text_model_type
        ),
        "zero_shot_single_cell_annotations": SingleCellZeroshotValidationScoreCalculator(
            batch_size=batch_size,
            transcriptome_tokenizer_type=transcriptome_model_type,
            tokenizer_name=text_model_type,
        ),
        "zero_shot_single_cell_annotations_well_studied_celltypes": SingleCellZeroshotValidationScoreCalculator(
            celltypes=TOP20_LUNG_LIVER_BLOOD_CELLTYPES,
            batch_size=batch_size,
            transcriptome_tokenizer_type=transcriptome_model_type,
            tokenizer_name=text_model_type,
        ),
    }
    return training_validation_functions
