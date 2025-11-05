"""
DataModule to be used for inference.

Inheriting from the training dataset (jointemb), but only generating embeddings for transcriptomes and only for .X (not for layers)

No need for prepare_data() etc. here
"""

from torch.utils.data import DataLoader
from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor
import anndata
from cellwhisperer.utils.processing import ensure_raw_counts_adata
from .jointemb import JointEmbedDataset
from pathlib import Path
from typing import Union


class CellxGenePreparationLoader(DataLoader):
    """
    Prepare the dataset.

    Only prepare the transcriptome data
    """

    def __init__(
        self,
        read_count_table: Union[anndata.AnnData, Path, str],
        transcriptome_processor="geneformer",
        transcriptome_processor_kwargs={},
        **kwargs
    ):
        """ """
        if isinstance(read_count_table, (str, Path)):
            read_count_table = anndata.read_h5ad(read_count_table)

        ensure_raw_counts_adata(read_count_table)

        self.transcriptome_processor = transcriptome_processor
        self.transcriptome_processor_kwargs = transcriptome_processor_kwargs

        # Load data and processor
        processor = TranscriptomeTextDualEncoderProcessor(
            self.transcriptome_processor,
            "dmis-lab/biobert-v1.1",  # unused
        )

        inputs = processor(
            text=None,
            transcriptomes=read_count_table,
            return_tensors="pt",
            padding="max_length",  # not sure if required (shouldn't actually)
        )

        dataset = JointEmbedDataset(
            inputs,
            orig_ids=read_count_table.obs.index.to_numpy(),
        )
        super().__init__(dataset, **kwargs)
