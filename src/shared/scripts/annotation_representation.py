import numpy as np
import json
import os

with open(snakemake.input.annotations) as f:
    annotations = json.load(f)

# Load model
from sentence_transformers import SentenceTransformer, models

word_embedding_model = models.Transformer("dmis-lab/biobert-v1.1")

pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_cls_token=True,
    pooling_mode_mean_tokens=False,
)
model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model], device="cuda"
)
model.eval()

# Get embeddings
embeddings = model.encode(
    list(annotations.values()), show_progress_bar=True, convert_to_numpy=True
)

# Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

np.savez(
    snakemake.output.representation,
    representation=embeddings,
    orig_ids=np.array(list(annotations.keys())),
)
