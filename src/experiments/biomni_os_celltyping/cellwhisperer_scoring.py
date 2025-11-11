import anndata
from cellwhisperer.utils.inference import score_transcriptomes_vs_texts
from cellwhisperer.utils.model_io import load_cellwhisperer_model
import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch.serialization
import cellwhisperer
import transformers

torch.serialization.add_safe_globals(
    [
        cellwhisperer.jointemb.config.TranscriptomeTextDualEncoderConfig,
        cellwhisperer.jointemb.geneformer_model.GeneformerConfig,
        transformers.models.bert.configuration_bert.BertConfig,
        cellwhisperer.jointemb.loss.config.LossConfig,
    ]
)

OUTPUT_FOLDER = "cellwhisperer_scoring_results"


pl_model, text_processor, transciptome_processor = load_cellwhisperer_model(
    "cellwhisperer_clip_v1.ckpt",
)


def load_h5ad(fn):
    adata = anndata.read_h5ad(fn)
    return adata


def score(adata, cell_types):
    score_results = score_transcriptomes_vs_texts(
        adata,
        cell_types,
        pl_model.model.discriminator.temperature.exp(),
        pl_model.model,
        transcriptome_processor=transciptome_processor,
        average_mode=None,
    )[0]
    # convert this tensor to a pandas dataframe
    scores_df = score_results.cpu().numpy().T
    scores_df = pd.DataFrame(
        scores_df,
        index=adata.obs_names,
        columns=cell_types,
    )
    return scores_df


with open("celltypes.json") as f:
    cell_types = json.load(f)

organ_to_file = {
    "brain": "brain_subsampled.h5ad",
    "breast": "breast_subsampled.h5ad",
    "gut": "gut.h5ad",
    "heart": "heart_subsampled.h5ad",
    "lung": "lung_subsampled.h5ad",
    "mouse_bonemarrow": "bonemarrow.h5ad",
    "mouse_kidney": "kidney.h5ad",
    "skin": "skin.h5ad",
}


def main():
    for organ, fn in organ_to_file.items():
        print(f"Scoring organ: {organ}")
        adata = load_h5ad(
            f"/dfs/user/hanchenw/BioAgentOS/data/singlecell/benchmark_human/{organ}/{fn}"
        )
        print("loaded adata")
        if adata.X.max() < 12:
            adata.X = np.expm1(adata.X).astype(int)

        print("scoring...")
        scores_df = score(adata, cell_types[organ])

        # get the argmax and store in adata.obs["celltype_cellwhisperer"]
        adata.obs["celltype_cellwhisperer"] = scores_df.idxmax(axis=1)

        print("saving...")
        path = Path(f"{OUTPUT_FOLDER}/cellwhisperer_annotated/{organ}.h5ad")
        path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(path)


if __name__ == "__main__":
    main()
