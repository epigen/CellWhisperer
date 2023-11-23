from pathlib import Path
import logging
import torch
from single_cellm.jointemb.model import TranscriptomeTextDualEncoderModel
from single_cellm.config import get_path, config
from single_cellm.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from single_cellm.validation.zero_shot.functions import get_scores_adatas_vs_text_list
from single_cellm.utils.cuda import get_device
from transformers import AutoTokenizer
import anndata


# Set up
device = get_device()
logger = logging.getLogger(__name__)


### Prepare the anndata object ###
logging.info("Loading anndata...")
adata = anndata.read_h5ad(
    get_path(["paths", "read_count_table"], dataset="tabula_sapiens_100_cells_per_type")
)
celltype_obs_colname = "cell_ontology_class"
counts_per_celltype = adata.obs.value_counts(celltype_obs_colname)
# Only process celltypes with at least 50 cells in the tabula sapiens dataset (There are a few with even less than 10)
celltypes_to_process = [
    x for x in counts_per_celltype[counts_per_celltype >= 50].index.values
]
celltypes_to_process = celltypes_to_process[:6]  # TODO
text_list = [f"Cell type: {x}" for x in celltypes_to_process]
adata_list = [
    adata[adata.obs[celltype_obs_colname] == celltype].copy()
    for celltype in celltypes_to_process
]
adata_dict = {
    celltype: adata_list[i] for i, celltype in enumerate(celltypes_to_process)
}


### Prepare the model ###
logging.info("Loading LLM embedding model...")
# TODO what should we use here instead of the hardcoded path?
geneformer_biogpt_model_path = Path(
    "~/projects/single-cellm/results/models/geneformer-biogpt"
).expanduser()
model = TranscriptomeTextDualEncoderModel.from_pretrained(
    geneformer_biogpt_model_path
).to(device)
text_tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
transcriptome_processor = GeneformerTranscriptomeProcessor(
    nproc=1, emb_label=model.transcriptome_model.config.emb_label
)

### Run the model and get the scores ###
result_dict, result_df = get_scores_adatas_vs_text_list(
    adata_dict_or_embedding_dict=adata_dict,
    model=model,
    device=device,
    text_tokenizer=text_tokenizer,
    transcriptome_processor=transcriptome_processor,
    text_list_or_text_embeds=None,
)  # automatically generates text_list from adata_dict
print(result_dict)

print(result_df)
