from .zero_shot.cancer_gene_essentiality import EvaluateCancerGeneEssentiality
from .zero_shot.single_cell_annotation import (
    SingleCellZeroshotValidationScoreCalculator,
    TOP20_LUNG_LIVER_BLOOD_CELLTYPES,
)
from .zero_shot.retrieval import RetrievalScoreCalculator
from torch.utils.data import DataLoader
from single_cellm.config import get_path, config
from single_cellm.jointemb.dataset.jointemb import JointEmbedDataModule


def initialize_validation_functions(
    batch_size: int,
    transcriptome_model_type: str,
    text_model_type: str,
    val_dataloader: DataLoader,
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
            tokenizer_name=text_model_type,
            transcriptome_tokenizer_type=transcriptome_model_type,
        ),
        # TODO: For deduplication, would need to provide the dataset name, anndata, or annotations. See src/validation/zero_shot/deduplicate.py
        "zero_shot_retrieval_validation_set": RetrievalScoreCalculator(val_dataloader),
    }

    # Add retrieval validation tests for the deduplicated validation-sets
    for name in config["retrieval_validation_sets"]:
        dm = JointEmbedDataModule(
            tokenizer=text_model_type,
            transcriptome_processor=transcriptome_model_type,
            dataset_name=name,
            batch_size=batch_size,
            train_fraction=0.0,
        )
        dm.prepare_data()
        dm.setup()
        training_validation_functions[name] = RetrievalScoreCalculator(
            dm.val_dataloader()
        )

    return training_validation_functions
