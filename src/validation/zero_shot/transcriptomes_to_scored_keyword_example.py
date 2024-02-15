from pathlib import Path
import logging
import torch
from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel
from cellwhisperer.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from cellwhisperer.validation.zero_shot.functions import (
    anndata_to_scored_keywords,
    formatted_text_from_df,
)
from cellwhisperer.misc.cuda import get_device
from cellwhisperer.config import get_path, config
from transformers import AutoTokenizer
import anndata
import subprocess
import yaml


### Example usage ###
use_immgen = False
run_example = True
if run_example:
    logger = logging.getLogger(__name__)

    # Prepare the anndata object
    logging.info("Loading anndata...")

    if use_immgen:
        # To read from csv:
        # adata = anndata.read_csv("https://sharehost.hms.harvard.edu/immgen/GSE227743/GSE227743_Normalized_Gene_count_table.csv",
        #     first_column_names=True,
        # ).T
        adata = anndata.read_h5ad(
            get_path(["paths", "read_count_table"], dataset="immgen")
        )
        if not "cell type" in adata.obs.columns:
            adata.obs["cell type"] = [x.split("#")[0] for x in adata.obs.index.values]
        if not "cell type rough" in adata.obs.columns:
            adata.obs["cell type rough"] = [
                x.split(".")[0] for x in adata.obs["cell type"].values
            ]
        adata = adata[adata.obs["cell type rough"] == "B"]
    else:
        adata = adata = anndata.read_h5ad(
            get_path(
                ["paths", "read_count_table"],
                dataset="tabula_sapiens_100_cells_per_type",
            )
        )
        adata = adata[adata.obs["cell_ontology_class"] == "mature nk t cell"]

    # Load the enrichr terms path (assumes this is already pre-computed, see cellwhisperer/src/validation/zero_shot/write_enrichr_terms.py):
    terms_json_path = get_path(["paths", "enrichr_terms_json"])

    # Model loading
    logging.info("Loading LLM embedding model...")
    # TODO load trained model checkpoint (see cellwhisperer_lightning.py)
    geneformer_biogpt_model_path = Path(
        "~/projects/cellwhisperer/results/models/geneformer-biogpt"
    ).expanduser()
    device = get_device()
    model = TranscriptomeTextDualEncoderModel.from_pretrained(
        geneformer_biogpt_model_path
    ).to(device)
    text_tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
    transcriptome_processor = GeneformerTranscriptomeProcessor(
        nproc=1, emb_label=model.transcriptome_model.config.emb_label
    )
    logging.info("Loading done")

    # Compute the top n keywords per cell type
    logging.info("Computing cell/keyword similarities ...")

    obs_cols = (
        ["cell_ontology_class"] if not use_immgen else ["cell type", "cell type rough"],
    )
    obs_cols = [
        c for c, t in adaptor.data.obs.dtypes.items() if isinstance(t, CategoricalDtype)
    ]
    additional_text_dict = {
        obs_col: adaptor.data.obs[obs_col].astype(str).unique().tolist()
        for obs_col in obs_cols
    }
    additional_text_dict["day_of_induction"] = ["10", "20"]
    additional_text_dict["treatment"] = ["dox"]

    similarity_scores_df = anndata_to_scored_keywords(
        adata_or_embedding=adata,
        model=model,
        terms=terms_json_path,
        transcriptome_processor=transcriptome_processor,
        text_tokenizer=text_tokenizer,
        device=device,
        average_mode="embeddings",
        batch_size=64,
        additional_text_dict=additional_text_dict,
        score_norm_method="zscore",
    )

    similarity_formatted_text = formatted_text_from_df(
        similarity_scores_df, n_top_per_term=5
    )
    print(similarity_formatted_text)
