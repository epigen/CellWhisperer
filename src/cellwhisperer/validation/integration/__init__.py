from typing import Optional, Any
from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel
from cellwhisperer.config import model_path_from_name
from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor
from cellwhisperer.validation.zero_shot.single_cell_annotation import (
    SingleCellDataSetForValidationScoring,
)
from cellwhisperer.utils.processing import adata_to_embeds
from .functions import eval_scib_metrics
import pandas as pd
from typing import Dict, Tuple


class SingleCellIntegrationScoreCalculator:
    def __init__(
        self,
        sc_dataset: SingleCellDataSetForValidationScoring,
        transcriptome_tokenizer_type: str = "geneformer",
        tokenizer_name: str = "biogpt",
        transcriptome_processor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Class to calculate integration scores for a single-cell dataset.
        Args:
            sc_dataset: a SingleCellDataSetForValidationScoring object.
            tokenizer_name: name of the tokenizer to use for the text. Must be a key in the config file.
            transcriptome_tokenizer_type: type of tokenizer to use for the transcriptome. Must be one of "geneformer", "scgpt" or "uce".
            transcriptome_processor_kwargs: kwargs to pass to the transcriptome processor. Default: None for geneformer, \
                {"gene_col":"gene_name"} for scgpt.
        """

        transcriptome_processor_kwargs = transcriptome_processor_kwargs or {}
        tokenizer_path = model_path_from_name(tokenizer_name)

        self.adata = sc_dataset.adata
        self.celltype_obs_colname = sc_dataset.celltype_obs_colname
        self.batch_obs_colname = sc_dataset.batch_obs_colname

        self.processor = TranscriptomeTextDualEncoderProcessor(
            transcriptome_tokenizer_type,
            tokenizer_path,
            **transcriptome_processor_kwargs,
        )

    def __call__(
        self, model: TranscriptomeTextDualEncoderModel
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Args:
            model: a trained model
        Returns:
            A tuple of:
            - A dictionary of the scib integration metrics for batch and celltype.
            - None
        """

        transcriptome_embeddings = adata_to_embeds(
            self.adata,
            model=model,
            transcriptome_processor=self.processor.transcriptome_processor,
        )

        self.adata.obsm[f"X_cellwhisperer"] = transcriptome_embeddings.cpu().numpy()

        integration_metrics = eval_scib_metrics(
            self.adata,
            label_key=self.celltype_obs_colname,
            batch_key=self.batch_obs_colname,
            embedding_key=f"X_cellwhisperer",
        )

        return (
            integration_metrics,
            None,
        )
