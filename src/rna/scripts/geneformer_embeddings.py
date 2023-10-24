from geneformer import EmbExtractor
from pathlib import Path
import numpy as np


embex = EmbExtractor(
    model_type="Pretrained",
    num_classes=0,
    emb_mode="cell",
    max_ncells=200,  # TODO
    emb_layer=-1,
    emb_label=["sample_name", "cell type rough", "cell type"],  # TODO
    labels_to_plot=["sample_name", "cell type rough", "cell type"],  # TODO
    forward_batch_size=1,
    nproc=1,
)
# TODO use temp directory here.
embs = embex.extract_embs(
    snakemake.input.model_path,
    snakemake.input["tokens_directory"],
    Path(snakemake.output[0]).parent,
    Path(snakemake.output[0]).stem,
    output_torch_embs=True,
)
# save embeddings as numpy array
np.savez_compressed(
    snakemake.output[0],
    ids=embs[0].sample_name.values.astype("U"),
    embeddings=embs[1].cpu().numpy(),
)
