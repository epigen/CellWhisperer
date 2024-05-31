from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel
from cellwhisperer.validation.zero_shot.single_cell_annotation import (
    SingleCellZeroshotValidationScoreCalculator,
)
from cellwhisperer.validation.zero_shot.retrieval import RetrievalScoreCalculator

from cellwhisperer.config import get_path, config
import torch
from cellwhisperer.jointemb.dataset.jointemb import JointEmbedDataModule
import torch
from cellwhisperer.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from cellwhisperer.validation.zero_shot.functions import (
    get_performance_metrics_transcriptome_vs_text,
)
from transformers import AutoTokenizer

# set the number of cores:
import os

os.environ["OMP_NUM_THREADS"] = "5"
# set cuda visible devices:
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#### Set up the model and tokenizers ####

model_dir = get_path(["model_name_path_map", "geneformer"])
config_transcriptome = {
    "model_directory": str(
        model_dir
    ),  # str(PROJECT_DIR / "resources" / "geneformer-12L-30M"),
    "model_type": "geneformer",
    "vocab_path": str(model_dir) + "vocab.json",
}
config_text = {"model_type": "biogpt"}
device = torch.device("cuda")
model = TranscriptomeTextDualEncoderModel.from_transcriptome_text_pretrained(
    transcriptome_model_name_or_path=str(model_dir),
    transcriptome_config=config_transcriptome,
    text_model_name_or_path="microsoft/biogpt",
    text_config=config_text,
)
model.to(device)
transcriptome_processor = GeneformerTranscriptomeProcessor(
    nproc=1, emb_label=model.transcriptome_model.config.emb_label
)

#### Test the SingleCellZeroshotValidationScoreCalculator ####

if True:
    # Should not average over cell types before predicting. Therefore 100 samples per celltype.
    score_calculator = SingleCellZeroshotValidationScoreCalculator(average_mode=None)
    (
        performance_metrics,
        performance_metrics_per_celltype_df,
    ) = score_calculator(model=model)

    # should average over cell types before predicting. Therefore only one sample per celltype.
    score_calculator = SingleCellZeroshotValidationScoreCalculator()
    (
        performance_metrics,
        performance_metrics_per_celltype_df,
    ) = score_calculator(model=model)


if False:
    # Same as above, now for all cells in the dataset (tabula sapiens max 100)
    score_calculator = SingleCellZeroshotValidationScoreCalculator(
        celltypes=None, average_mode=None
    )
    (
        performance_metrics,
        performance_metrics_per_celltype_df,
    ) = score_calculator(model=model)

    # Same as above, but now for all cells in the full dataset (complete tabula sapiens)
    score_calculator = SingleCellZeroshotValidationScoreCalculator(
        celltypes=None,
        average_mode=None,
        dataset="tabula_sapiens",
    )
    performance_metrics, performance_metrics_per_celltype_df = score_calculator(
        model=model
    )

#### Test the RetrievalScoreCalculator ####

if True:
    dm = JointEmbedDataModule(
        tokenizer="biogpt",
        transcriptome_processor="geneformer",
        dataset_name="human_disease",
        batch_size=32,
        train_fraction=0.0,
    )
    dm.prepare_data()
    dm.setup()
    retrieval_calculator = RetrievalScoreCalculator(dm.val_dataloader())
    (
        performance_metrics,
        performance_metrics_per_transcriptome_df,
    ) = retrieval_calculator(model=model)

    print(performance_metrics)
    print(performance_metrics_per_transcriptome_df)

#### Test with random data ####

if True:
    # create a torch tensor: 1000 cells x 512 dims
    random_tensor = torch.rand(1000, 512, device=device)
    ### Run the model and get the scores ###
    (
        performance_metrics,
        performance_metrics_per_transcriptome_df,
    ) = get_performance_metrics_transcriptome_vs_text(
        transcriptome_input=random_tensor,
        model=model,
        transcriptome_processor=transcriptome_processor,
        text_list_or_text_embeds=random_tensor,
        correct_text_idx_per_transcriptome=list(range(random_tensor.shape[0])),
        average_mode=None,
    )
    print(performance_metrics)  # All 1.

    # create a tensor where always 10 cells are the same. 15 cell types
    transcriptome_tensor = torch.zeros(150, 512, device=device)
    text_tensor = torch.zeros(15, 512, device=device)

    transcriptome_annotations = []

    for celltype_idx, i in enumerate(range(0, 150, 10)):
        tensor_this_celltype = torch.rand(1, 512, device=device)
        transcriptome_tensor[i : i + 10, :] = tensor_this_celltype
        text_tensor[celltype_idx, :] = tensor_this_celltype
        transcriptome_annotations += [celltype_idx for _ in range(10)]

    # add a small amount of randomness to the text tensor
    text_tensor += torch.rand(15, 512, device=device) * 0.1

    # make cell type #13 the same as cell type #14
    transcriptome_tensor[130:140, :] = transcriptome_tensor[140:150, :]

    (
        performance_metrics,
        performance_metrics_per_celltype_df,
    ) = get_performance_metrics_transcriptome_vs_text(
        transcriptome_input=transcriptome_tensor,
        model=model,
        transcriptome_processor=transcriptome_processor,
        text_list_or_text_embeds=text_tensor,
        correct_text_idx_per_transcriptome=transcriptome_annotations,
        grouping_keys=grouping_keys,
        average_mode="embeddings",
    )

    # Looks good: scores are on the diagonal, except for cell #13, which matches with text for 14
    # The scores are 1 for all cell types except 13 and 14.

    print(
        performance_metrics_per_celltype_df
    )  # Works well - scores are on the diagonal, except for cell #13, which matches with text for 14
