from .zero_shot.cancer_gene_essentiality import EvaluateCancerGeneEssentiality
from .zero_shot.single_cell_annotation import (
    SingleCellDataSetForValidationScoring,
    SingleCellZeroshotValidationScoreCalculator,
    TOP20_LUNG_LIVER_BLOOD_CELLTYPES,
)
from .integration import SingleCellIntegrationScoreCalculator
from .zero_shot.retrieval import RetrievalScoreCalculator
from torch.utils.data import DataLoader
from cellwhisperer.config import get_path, config
from cellwhisperer.jointemb.dataset.jointemb import JointEmbedDataModule
from typing import Optional


def initialize_validation_functions(
    batch_size: int,
    transcriptome_model_type: str,
    text_model_type: str,
    val_dataloader: Optional[DataLoader] = None,
):
    tabsap_sc_dataset = SingleCellDataSetForValidationScoring(cell_number_threshold_per_celltype=100)
    tabsap_wellstudied_sc_dataset = SingleCellDataSetForValidationScoring(
        celltypes=TOP20_LUNG_LIVER_BLOOD_CELLTYPES
    )

    training_validation_functions = {
        # "zshot_cancer_gene_essentiality": EvaluateCancerGeneEssentiality(
        #     batch_size, transcriptome_model_type, text_model_type
        # ),
        "zshot_TabSap_celltype_lvl": SingleCellZeroshotValidationScoreCalculator(
            sc_dataset=tabsap_sc_dataset,
            batch_size=batch_size,
            transcriptome_tokenizer_type=transcriptome_model_type,
            tokenizer_name=text_model_type,
        ),
        "zshot_TabSapWellStudied_celltype_lvl": SingleCellZeroshotValidationScoreCalculator(
            sc_dataset=tabsap_wellstudied_sc_dataset,
            batch_size=batch_size,
            tokenizer_name=text_model_type,
            transcriptome_tokenizer_type=transcriptome_model_type,
        ),
        "zshot_TabSap_cell_lvl": SingleCellZeroshotValidationScoreCalculator(
            sc_dataset=tabsap_sc_dataset,
            batch_size=batch_size,
            transcriptome_tokenizer_type=transcriptome_model_type,
            tokenizer_name=text_model_type,
            average_mode=None,
        ),
        "zshot_TabSapWellStudied_cell_lvl": SingleCellZeroshotValidationScoreCalculator(
            sc_dataset=tabsap_wellstudied_sc_dataset,
            batch_size=batch_size,
            tokenizer_name=text_model_type,
            transcriptome_tokenizer_type=transcriptome_model_type,
            average_mode=None,
        ),

        "integration_TabSapWellStudied": SingleCellIntegrationScoreCalculator(
            sc_dataset=tabsap_wellstudied_sc_dataset,
            tokenizer_name=text_model_type,
            transcriptome_tokenizer_type=transcriptome_model_type
        ),
        "integration_TabSap": SingleCellIntegrationScoreCalculator(
            sc_dataset=tabsap_sc_dataset,
            tokenizer_name=text_model_type,
            transcriptome_tokenizer_type=transcriptome_model_type
        ),

    }
    if val_dataloader is not None:
        # TODO: For deduplication, would need to provide the dataset name, anndata, or annotations. See src/validation/zero_shot/deduplicate.py
        training_validation_functions[
            "zshot_retrieval_validation_set"
        ] = RetrievalScoreCalculator(val_dataloader)

    # Add retrieval validation tests for the deduplicated validation-sets
    for name in config["retrieval_validation_sets"]:
        dm = JointEmbedDataModule(
            tokenizer=text_model_type,
            transcriptome_processor=transcriptome_model_type,
            dataset_names=name,
            batch_size=batch_size,
            train_fraction=0.0,
        )
        dm.prepare_data()
        dm.setup()
        training_validation_functions[name] = RetrievalScoreCalculator(
            dm.val_dataloader()
        )

    return training_validation_functions
