from pathlib import Path
import logging
import torch
from single_cellm.jointemb.model import TranscriptomeTextDualEncoderModel
from single_cellm.config import get_path, config
from single_cellm.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from single_cellm.validation.zero_shot.transcriptomes_to_scored_keywords import get_scores_adatas_vs_text_list
from single_cellm.utils.cuda import set_freest_gpu_as_device
from transformers import AutoTokenizer
import anndata
import numpy as np
np.random.seed(42)

class SingleCellZeroshotValidationScoreCalculator:
    def __init__(self, n_celltypes=10, cell_number_threshold_per_celltype=100, dataset="tabula_sapiens_100_cells_per_type",
                 celltype_obs_colname="cell_ontology_class",
                 prefix_for_text_embeddings="Transcriptome of a ",
                 suffix_for_text_embeddings="",
                 nproc_transcriptome_processor=1,
                 device=None,
                 logger=None,
                 tokenizer_name="microsoft/biogpt"):
        """
        Class to calculate zero-shot validation scores for a single-cell dataset.
        Args:
            n_celltypes: number of celltypes to process. This many celltypes will be randomly sampled from the dataset.
            cell_number_threshold_per_celltype: only celltypes with at least this number of cells will be processed.
            dataset: name of the dataset to process. Must be a key in the config file.
            celltype_obs_colname: name of the column in the adata.obs dataframe that contains the celltype labels.
            prefix_for_text_embeddings: prefix to add to the celltype name to generate the text to embed.
            suffix_for_text_embeddings: suffix to add to the celltype name to generate the text to embed.
            nproc_transcriptome_processor: number of processes to use for the transcriptome processor.
            device: device to use for the model. If None, will use the first available GPU if available, otherwise CPU.
            logger: logger to use. If None, will use the logger for this module.
        """

        self.n_celltypes = n_celltypes
        self.cell_number_threshold_per_celltype = cell_number_threshold_per_celltype
        self.dataset = dataset
        self.celltype_obs_colname = celltype_obs_colname
        self.prefix_for_text_embeddings = prefix_for_text_embeddings
        self.suffix_for_text_embeddings = suffix_for_text_embeddings
        self.nproc_transcriptome_processor = nproc_transcriptome_processor
        self.device = device
        self.logger = logger
        self.tokenizer_name = tokenizer_name

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # set_freest_gpu_as_device()

        if self.logger is None:
            self.logger = logging.getLogger(__name__)

        self.logger.info("Loading anndata...")
        self.adata = anndata.read_h5ad(
            get_path(["paths", "read_count_table"], dataset=self.dataset)
        )
        
        self.counts_per_celltype = self.adata.obs.value_counts(self.celltype_obs_colname)
        self.celltypes_to_process = [x for x in self.counts_per_celltype[self.counts_per_celltype>=self.cell_number_threshold_per_celltype].index.values]
        assert len(self.celltypes_to_process)>=self.n_celltypes, f"Only {len(self.celltypes_to_process)} celltypes have at least {self.cell_number_threshold_per_celltype} cells, but {self.n_celltypes} celltypes were requested."
        np.random.shuffle(self.celltypes_to_process)
        self.celltypes_to_process = self.celltypes_to_process[:self.n_celltypes]
        self.text_list = [f"{self.prefix_for_text_embeddings}{x}{self.suffix_for_text_embeddings}" for x in self.celltypes_to_process]
        self.adata_list = [self.adata[self.adata.obs[self.celltype_obs_colname] == celltype].copy() for celltype in self.celltypes_to_process]
        self.adata_dict = {celltype: self.adata_list[i] for i, celltype in enumerate(self.celltypes_to_process)}


        self.text_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        self.transcriptome_processor = GeneformerTranscriptomeProcessor(
            nproc=self.nproc_transcriptome_processor, emb_label=[]) # I think it's ok to not have emb_labels here, but I'm not sure
        

    def get_scores(self, model):

        result_dict = get_scores_adatas_vs_text_list(adata_dict_or_embedding_dict=self.adata_dict,
                                                model=model,
                                                device=self.device,
                                                text_tokenizer=self.text_tokenizer,
                                                transcriptome_processor=self.transcriptome_processor,
                                                text_list_or_text_embeds=None) # automatically generates text_list from adata_dict
        return result_dict

