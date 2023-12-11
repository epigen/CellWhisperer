from typing import Union, Iterable, Optional, Any
import logging
from single_cellm.jointemb.model import TranscriptomeTextDualEncoderModel
from single_cellm.config import get_path
from single_cellm.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from single_cellm.jointemb.scgpt_model import ScGPTTranscriptomeProcessor
from single_cellm.validation.zero_shot.functions import get_scores_adatas_vs_text_list

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
        celltypes: Union[int, Iterable[str]] = 10,
        cell_number_threshold_per_celltype: int = 100,
        dataset: str = "tabula_sapiens_100_cells_per_type",
        celltype_obs_colname: str = "cell_ontology_class",
        prefix_for_text_embeddings="Sample of a ",
        suffix_for_text_embeddings: str = "",
        nproc_transcriptome_processor: str = 1,
        logger: Optional[Any] = None,
        tokenizer_name: str = "microsoft/biogpt",
        transcriptome_tokenizer_type="geneformer",
        transcriptome_processor_kwargs=None,
        batch_size: int = 32,
    ):
        """
        Class to calculate zero-shot validation scores for a single-cell dataset.
        Args:
            celltypes: number of celltypes to process. \
                If int: This many celltypes will be randomly sampled from the dataset.
                If list: This list of celltypes will be processed.
            cell_number_threshold_per_celltype: only celltypes with at least this number of cells will be processed.
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
        """

        self.cell_number_threshold_per_celltype = cell_number_threshold_per_celltype
        self.dataset = dataset
        self.celltype_obs_colname = celltype_obs_colname
        self.prefix_for_text_embeddings = prefix_for_text_embeddings
        self.suffix_for_text_embeddings = suffix_for_text_embeddings
        self.nproc_transcriptome_processor = nproc_transcriptome_processor
        self.logger = logger
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        self.transcriptome_tokenizer_type = transcriptome_tokenizer_type
        if transcriptome_processor_kwargs is None:
            if self.transcriptome_tokenizer_type == "scgpt":
                self.transcriptome_processor_kwargs = {"gene_col": "gene_name"}
            else:
                self.transcriptome_processor_kwargs = {}
        else:
            self.transcriptome_processor_kwargs = transcriptome_processor_kwargs

        if self.logger is None:
            self.logger = logging.getLogger(__name__)

        self.logger.info("Loading anndata...")
        self.adata = anndata.read_h5ad(
            get_path(["paths", "read_count_table"], dataset=self.dataset)
        )

        if isinstance(celltypes, int):
            self.counts_per_celltype = self.adata.obs.value_counts(
                self.celltype_obs_colname
            )
            self.celltypes_to_process = [
                x
                for x in self.counts_per_celltype[
                    self.counts_per_celltype >= self.cell_number_threshold_per_celltype
                ].index.values
            ]
            assert (
                len(self.celltypes_to_process) >= celltypes
            ), f"Only {len(self.celltypes_to_process)} celltypes have at least {self.cell_number_threshold_per_celltype} cells, but {celltypes} celltypes were requested."
            np.random.shuffle(self.celltypes_to_process)
            self.celltypes_to_process = self.celltypes_to_process[:celltypes]
        elif isinstance(celltypes, Iterable):
            self.celltypes_to_process = celltypes
        self.text_list = [
            f"{self.prefix_for_text_embeddings}{x}{self.suffix_for_text_embeddings}"
            for x in self.celltypes_to_process
        ]
        # REFACTOR: use a single adata object with celltype within `obs` DataFrame
        self.adata_list = [
            self.adata[self.adata.obs[self.celltype_obs_colname] == celltype].copy()
            for celltype in self.celltypes_to_process
        ]
        self.adata_dict = {
            celltype: self.adata_list[i]
            for i, celltype in enumerate(self.celltypes_to_process)
        }

        self.text_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        if transcriptome_tokenizer_type == "geneformer":
            self.transcriptome_processor = GeneformerTranscriptomeProcessor(
                nproc=self.nproc_transcriptome_processor,
                emb_label=[],
                **self.transcriptome_processor_kwargs,
            )  # I think it's ok to not have emb_labels here, but I'm not sure
        elif transcriptome_tokenizer_type == "scgpt":
            self.transcriptome_processor = ScGPTTranscriptomeProcessor(
                nproc=self.nproc_transcriptome_processor,
                **self.transcriptome_processor_kwargs,
            )
        else:
            ValueError(
                f"transcriptome_tokenizer_type must be one of 'geneformer' or 'scgpt', but is {transcriptome_tokenizer_type}."
            )

    def __call__(self, model) -> Tuple[Dict[str, float], pd.DataFrame]:
        result_dict, result_df = get_scores_adatas_vs_text_list(
            adata_dict_or_embedding_dict=self.adata_dict,
            model=model,
            text_tokenizer=self.text_tokenizer,
            transcriptome_processor=self.transcriptome_processor,
            text_list_or_text_embeds=None,
            batch_size=self.batch_size,
        )  # automatically generates text_list from adata_dict
        return result_dict, result_df
