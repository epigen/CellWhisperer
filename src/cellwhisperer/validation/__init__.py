from .zero_shot.cancer_gene_essentiality import EvaluateCancerGeneEssentiality
from .zero_shot.single_cell_annotation import (
    SingleCellDataSetForValidationScoring,
    SingleCellZeroshotValidationScoreCalculator,
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
    image_model_type: str,
):
    tabsap_sc_dataset = SingleCellDataSetForValidationScoring(
        cell_number_threshold_per_celltype=100
    )
    tabsap_wellstudied_sc_dataset = SingleCellDataSetForValidationScoring(
        celltypes=config["top20_lung_liver_blood_celltypes"]
    )
    
    # Lung tissue datasets for validation
    lung_tissue_dataset = SingleCellDataSetForValidationScoring(
        dataset="lung_tissue",
        celltype_obs_colname="cell_type_annotations",
        cell_number_threshold_per_celltype=200,
        auto_create_batch_obs_colname=False,  # Spatial data doesn't have donor/method structure
        use_image_data=True,  # Enable image data usage for lung tissue
    )
    
    lung_tissue_region_dataset = SingleCellDataSetForValidationScoring(
        dataset="lung_tissue",
        celltype_obs_colname="region_type_expert_annotation",
        cell_number_threshold_per_celltype=200,
        auto_create_batch_obs_colname=False,  # Spatial data doesn't have donor/method structure
        use_image_data=True,  # Enable image data usage for lung tissue
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
            image_processor=image_model_type,
        ),
        "zshot_TabSapWellStudied_celltype_lvl": SingleCellZeroshotValidationScoreCalculator(
            sc_dataset=tabsap_wellstudied_sc_dataset,
            batch_size=batch_size,
            tokenizer_name=text_model_type,
            transcriptome_tokenizer_type=transcriptome_model_type,
            image_processor=image_model_type,
        ),
        "zshot_TabSap_cell_lvl": SingleCellZeroshotValidationScoreCalculator(
            sc_dataset=tabsap_sc_dataset,
            batch_size=batch_size,
            transcriptome_tokenizer_type=transcriptome_model_type,
            tokenizer_name=text_model_type,
            average_mode=None,
            image_processor=image_model_type,
        ),
        "zshot_TabSapWellStudied_cell_lvl": SingleCellZeroshotValidationScoreCalculator(
            sc_dataset=tabsap_wellstudied_sc_dataset,
            batch_size=batch_size,
            tokenizer_name=text_model_type,
            transcriptome_tokenizer_type=transcriptome_model_type,
            average_mode=None,
            image_processor=image_model_type,
        ),
        "integration_TabSapWellStudied": SingleCellIntegrationScoreCalculator(
            sc_dataset=tabsap_wellstudied_sc_dataset,
            tokenizer_name=text_model_type,
            transcriptome_tokenizer_type=transcriptome_model_type,
            image_processor=image_model_type,
        ),
        "integration_TabSap": SingleCellIntegrationScoreCalculator(
            sc_dataset=tabsap_sc_dataset,
            tokenizer_name=text_model_type,
            transcriptome_tokenizer_type=transcriptome_model_type,
            image_processor=image_model_type,
        ),
        #"zshot_LungTissue_region_lvl": SingleCellZeroshotValidationScoreCalculator(
        #    sc_dataset=lung_tissue_region_dataset,
        #    batch_size=batch_size,
        #    transcriptome_tokenizer_type=transcriptome_model_type,
        #    tokenizer_name=text_model_type,
        #    image_processor=image_model_type,
        #),

    }

    # Add retrieval tests for the deduplicated validation-sets
    for name in config["retrieval_deduplicated_sets"]:
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
