from typing import Union, Iterable, Optional, Any
from pathlib import Path
import logging
from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel
from cellwhisperer.config import get_path, model_path_from_name
from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor
from cellwhisperer.validation.zero_shot.functions import (
    get_performance_metrics_transcriptome_vs_text,
)
from cellwhisperer.utils.processing import adata_to_embeds
import anndata
import numpy as np
import pandas as pd
from typing import Dict, Tuple


class SingleCellDataSetForValidationScoring:
    def __init__(
        self,
        celltypes: Optional[Union[int, Iterable[str]]] = None,
        cell_number_threshold_per_celltype: Optional[int] = None,
        dataset: Union[str, Path] = "tabula_sapiens_100_cells_per_type",
        celltype_obs_colname: str = "cell_ontology_class",
        batch_obs_colname="batch",
        auto_create_batch_obs_colname: bool = True,
        logger: Optional[Any] = None,
    ):
        """
        Class to process a single-cell dataset and prepare it for validation scoring.
        Args:
            celltypes: number of celltypes to process. \
                If int: This many celltypes will be randomly sampled from the dataset.
                If list: This list of celltypes will be processed.
                If None: All celltypes in the dataset will be processed (after applying cell_number_threshold_per_celltype, if set).
            cell_number_threshold_per_celltype: only celltypes with at least this number of cells will be processed.
            dataset: if `str` then name of the dataset to process (must be a key in the config file). If `Path` then path to the anndata file.
            celltype_obs_colname: name of the column in the adata.obs dataframe that contains the celltype labels.
            batch_obs_colname: name of the column in the adata.obs dataframe that contains the batch labels.
            auto_create_batch_obs_colname: If true, set adata.obs["batch"] = (adata.obs["donor"].astype(str) + "_" + adata.obs["method"].astype(str))
            logger: logger to use. If None, will use the logger for this module.

        """

        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Loading anndata...")
        if isinstance(dataset, str):
            dataset_path = get_path(["paths", "read_count_table"], dataset=dataset)
        else:
            dataset_path = dataset
            assert isinstance(dataset, Path)

        self.adata = anndata.read_h5ad(dataset_path)

        # NOTE: X should anyways be raw counts
        if "raw_counts" in self.adata.layers:
            self.adata.X = self.adata.layers["raw_counts"]

        # First, filter out celltypes with less than cell_number_threshold_per_celltype cells, if necessary
        if cell_number_threshold_per_celltype is not None:
            self.counts_per_celltype = self.adata.obs.value_counts(celltype_obs_colname)
            self.celltypes_to_process = [
                x
                for x in self.counts_per_celltype[
                    self.counts_per_celltype >= cell_number_threshold_per_celltype
                ].index.values
            ]
        else:
            self.celltypes_to_process = self.adata.obs[celltype_obs_colname].unique()

        if isinstance(celltypes, int):  # Randomly sample celltypes
            assert (
                len(self.celltypes_to_process) >= celltypes
            ), f"Only {len(self.celltypes_to_process)} celltypes have at least {cell_number_threshold_per_celltype} cells, but {celltypes} celltypes were requested."
            np.random.seed(42)
            np.random.shuffle(self.celltypes_to_process)
            self.celltypes_to_process = self.celltypes_to_process[:celltypes]
        elif isinstance(celltypes, Iterable):
            assert all(
                [x in self.celltypes_to_process for x in celltypes]
            ), "Not all celltypes in argument celltypes are present in the dataset after filtering via cell_number_threshold_per_celltype."
            self.celltypes_to_process = celltypes
        elif celltypes is None:
            pass
        else:
            raise ValueError("celltypes must be an int, a list of strings, or None.")

        self.adata = self.adata[
            self.adata.obs[celltype_obs_colname].isin(self.celltypes_to_process), :
        ].copy()  # subset the adata according to the celltypes to process

        if auto_create_batch_obs_colname == True:
            self.adata.obs["batch"] = (
                self.adata.obs["donor"].astype(str)
                + "_"
                + self.adata.obs["method"].astype(str)
            )

        self.celltype_obs_colname = celltype_obs_colname
        self.batch_obs_colname = batch_obs_colname


class SingleCellZeroshotValidationScoreCalculator:
    def __init__(
        self,
        sc_dataset: SingleCellDataSetForValidationScoring,
        prefix_for_text_embeddings: str = "Sample of a ",
        suffix_for_text_embeddings: str = "",
        nproc_transcriptome_processor: int = 0,
        tokenizer_name: str = "biogpt",
        transcriptome_tokenizer_type: str = "geneformer",
        transcriptome_processor_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 32,
        average_mode: Optional[str] = "embeddings",
    ):
        """
        Class to calculate zero-shot validation scores for a single-cell dataset.
        Args:
            sc_dataset: a SingleCellDataSetForValidationScoring object.
            prefix_for_text_embeddings: prefix to add to the celltype name to generate the text to embed.
            suffix_for_text_embeddings: suffix to add to the celltype name to generate the text to embed.
            nproc_transcriptome_processor: number of processes to use for the transcriptome processor.
            tokenizer_name: name of the tokenizer to use for the text. Must be a key in the config file.
            transcriptome_tokenizer_type: type of tokenizer to use for the transcriptome. Must be one of "geneformer" or "scgpt".
            transcriptome_processor_kwargs: kwargs to pass to the transcriptome processor. Default: None for geneformer, \
                {"gene_col":"gene_name"} for scgpt.
            batch_size: batch size to use for the forward pass through the model and for the score calculation.
            average_mode: how to average the transcriptome embeddings. Must be one of "cells", "embeddings", or None.
        """

        self.batch_size = batch_size
        self.average_mode = average_mode

        tokenizer_path = model_path_from_name(tokenizer_name)
        transcriptome_processor_kwargs = transcriptome_processor_kwargs or {}

        self.adata = sc_dataset.adata
        celltype_obs_colname = sc_dataset.celltype_obs_colname
        celltypes_to_process = sc_dataset.celltypes_to_process

        self.text_list = [
            f"{prefix_for_text_embeddings}{x}{suffix_for_text_embeddings}"
            for x in celltypes_to_process
        ]

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
            nproc=nproc_transcriptome_processor,
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
            - A dictionary containing macro-averaged precision, recall (at k=1,5,10,50), accuracy, f1, and rocauc, \
                plus the scib integration metrics for batch and celltype.
            - A dataframe with cell type as rows and the above performance metrics for those cell types as columns.
        """

        transcriptome_embeddings = adata_to_embeds(
            self.adata,
            model=model,
            transcriptome_processor=self.processor.transcriptome_processor,
        )

        (
            performance_metrics,
            performance_metrics_per_celltype_df,
        ) = get_performance_metrics_transcriptome_vs_text(
            transcriptome_input=transcriptome_embeddings,
            model=model,
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
