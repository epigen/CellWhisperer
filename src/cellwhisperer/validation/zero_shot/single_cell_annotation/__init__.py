from typing import Union, Iterable, Optional, Any
import logging
from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel
from cellwhisperer.config import get_path, model_path_from_name
from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor
from cellwhisperer.validation.zero_shot.functions import (
    get_performance_metrics_transcriptome_vs_text,
)

from transformers import AutoTokenizer
import anndata
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Created by running this on the full tabula sapiens dataset: list(adata.obs[adata.obs["organ_tissue"].isin(["Liver","Lung","Blood"])]["cell_ontology_class"].value_counts().iloc[:20].index)
TOP20_LUNG_LIVER_BLOOD_CELLTYPES = [
    "macrophage",
    "erythrocyte",
    "monocyte",
    "type ii pneumocyte",
    "classical monocyte",
    "neutrophil",
    "cd4-positive, alpha-beta t cell",
    "nk cell",
    "naive b cell",
    "basal cell",
    "cd8-positive, alpha-beta t cell",
    "hepatocyte",
    "cd8-positive, alpha-beta cytokine secreting effector t cell",
    "club cell",
    "non-classical monocyte",
    "capillary endothelial cell",
    "cd4-positive, alpha-beta memory t cell",
    "memory b cell",
    "respiratory goblet cell",
    "basophil",
]


class SingleCellZeroshotValidationScoreCalculator:
    def __init__(
        self,
        celltypes: Union[int, Iterable[str], None] = 10,
        cell_number_threshold_per_celltype: int = 100,
        dataset: str = "tabula_sapiens_100_cells_per_type",
        celltype_obs_colname: str = "cell_ontology_class",
        prefix_for_text_embeddings: str = "Sample of a ",
        suffix_for_text_embeddings: str = "",
        nproc_transcriptome_processor: int = 0,  # TODO link to CLI nproc argument
        logger: Optional[Any] = None,
        tokenizer_name: str = "biogpt",
        transcriptome_tokenizer_type: str = "geneformer",
        transcriptome_processor_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 32,
        average_mode: Optional[str] = "embeddings",
    ):
        """
        Class to calculate zero-shot validation scores for a single-cell dataset.
        Args:
            celltypes: number of celltypes to process. \
                If int: This many celltypes will be randomly sampled from the dataset.
                If list: This list of celltypes will be processed.
                If None: All celltypes in the dataset will be processed.
            cell_number_threshold_per_celltype: only celltypes with at least this number of cells will be processed. \
                Only used if celltypes is an int.
            dataset: name of the dataset to process. Must be a key in the config file.
            celltype_obs_colname: name of the column in the adata.obs dataframe that contains the celltype labels.
            prefix_for_text_embeddings: prefix to add to the celltype name to generate the text to embed.
            suffix_for_text_embeddings: suffix to add to the celltype name to generate the text to embed.
            nproc_transcriptome_processor: number of processes to use for the transcriptome processor.
            logger: logger to use. If None, will use the logger for this module.
            tokenizer_name: name of the tokenizer to use for the text. Must be a key in the config file.
            transcriptome_tokenizer_type: type of tokenizer to use for the transcriptome. Must be one of "geneformer" or "scgpt".
            transcriptome_processor_kwargs: kwargs to pass to the transcriptome processor. Default: None for geneformer, \
                {"gene_col":"gene_name"} for scgpt.
            batch_size: batch size to use for the forward pass through the model and for the score calculation.
            average_mode: how to average the transcriptome embeddings. Must be one of "cells", "embeddings", or None.
        """

        self.nproc_transcriptome_processor = (
            nproc_transcriptome_processor  # TODO make use of it
        )
        self.logger = logger or logging.getLogger(__name__)
        self.batch_size = batch_size
        self.average_mode = average_mode

        tokenizer_path = model_path_from_name(tokenizer_name)
        transcriptome_processor_kwargs = transcriptome_processor_kwargs or {}

        self.logger.info("Loading anndata...")
        self.adata = anndata.read_h5ad(
            get_path(["paths", "read_count_table"], dataset=dataset)
        )

        if isinstance(celltypes, int):
            self.counts_per_celltype = self.adata.obs.value_counts(celltype_obs_colname)
            self.celltypes_to_process = [
                x
                for x in self.counts_per_celltype[
                    self.counts_per_celltype >= cell_number_threshold_per_celltype
                ].index.values
            ]
            assert (
                len(self.celltypes_to_process) >= celltypes
            ), f"Only {len(self.celltypes_to_process)} celltypes have at least {cell_number_threshold_per_celltype} cells, but {celltypes} celltypes were requested."
            np.random.seed(42)
            np.random.shuffle(self.celltypes_to_process)
            self.celltypes_to_process = self.celltypes_to_process[:celltypes]
        elif isinstance(celltypes, Iterable):
            self.celltypes_to_process = celltypes
        elif celltypes is None:
            self.celltypes_to_process = self.adata.obs[celltype_obs_colname].unique()

        self.text_list = [
            f"{prefix_for_text_embeddings}{x}{suffix_for_text_embeddings}"
            for x in self.celltypes_to_process
        ]
        self.adata = self.adata[
            self.adata.obs[celltype_obs_colname].isin(self.celltypes_to_process), :
        ].copy()  # subset the adata according to the celltypes to process
        self.annotation = list(self.adata.obs[celltype_obs_colname].values)

        self.correct_text_idx_per_transcriptome = [
            self.text_list.index(
                f"{prefix_for_text_embeddings}{x}{suffix_for_text_embeddings}"
            )
            for x in self.annotation
        ]

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
            - A dictionary containing macro-averaged precision, recall (at k=1,5,10,50), accuracy, f1, and rocauc. \
            - A dataframe with cell type as rows and the above performance metrics for those cell types as columns.
        """
        (
            performance_metrics,
            performance_metrics_per_celltype_df,
        ) = get_performance_metrics_transcriptome_vs_text(
            transcriptome_input=self.adata,
            model=model,
            text_tokenizer=self.processor.tokenizer,
            transcriptome_processor=self.processor.transcriptome_processor,
            correct_text_idx_per_transcriptome=self.correct_text_idx_per_transcriptome,
            text_list_or_text_embeds=self.text_list,
            grouping_keys=self.annotation,
            average_mode=self.average_mode,
            batch_size=self.batch_size,
        )
        return (
            performance_metrics,
            performance_metrics_per_celltype_df,
        )
