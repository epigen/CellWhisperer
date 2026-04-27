---
name: cellwhisperer
description: |
  Use when the user needs to work with single-cell RNA-seq data using CellWhisperer: processing datasets for the interactive web app, scoring/annotating cells' types and states with free-text queries, or loading CellWhisperer as a Python library for local inference. Triggers on: "cellwhisperer", "cell type annotation", "scRNA-seq analysis", "single-cell scoring", "cellxgene", or any request involving transcriptome-to-text similarity scoring (e.g. when the user asks for specific attributes/properties of cells, including their type and state).
---

# CellWhisperer

CellWhisperer is a multimodal AI model combining transcriptomics with natural language to enable intuitive interaction with scRNA-seq datasets. Published in [Nature Biotechnology](https://doi.org/10.1038/s41587-025-02857-9).

This skill provides three capabilities:

1. **End-to-end analysis** — Prepare an h5ad dataset, process it through the CellWhisperer pipeline, and launch the interactive cellxgene web app.
2. **API-based cell scoring** — Query the hosted CellWhisperer API at `cellwhisperer.bocklab.org` to embed texts and score/annotate cells on demand, without local model installation.
3. **Local library usage** — Install CellWhisperer as a Python library and use it programmatically for model loading, embedding, and scoring.

## Installation (for Claude Code users)

```bash
# generally prevent auto-update for your safety
claude plugin marketplace add epigen/cellwhisperer@v0.1.0
claude plugin install cellwhisperer@cellwhisperer
```

After installing, restart Claude Code or run `/reload-plugins`. The skill becomes available as `/cellwhisperer` or is invoked automatically when CellWhisperer-related tasks are detected.

## Project setup

CellWhisperer uses [pixi](https://pixi.sh) for environment management.

```bash
git clone git@github.com:epigen/cellwhisperer.git --recurse-submodules
cd cellwhisperer
```

All commands below should be run from the CellWhisperer project root using `pixi run`.

Before starting, read the project README for full context:
- `README.md` — installation, dataset format, web app launch, paper reproduction

---

## Feature 1: End-to-End scRNA-seq Analysis

Goal: take a user's h5ad file from raw counts to an interactive CellWhisperer-powered cellxgene browser.

### Step 1: Prepare the dataset

Place the h5ad file at `resources/<dataset_name>/read_count_table.h5ad`.

**Requirements** (validate before proceeding):
- Raw integer read counts in `.X` or `.layers["counts"]` (int32, no NaN)
- `.var` must have a unique index and a `gene_name` column with gene symbols
- Recommended: provide `ensembl_id` in `.var` (computed if missing)
- Recommended: filter cells with <100 genes expressed
- Use `categorical` dtype for categorical `.obs` columns
- 2D embeddings in `.obsm` must be `np.ndarray` (not DataFrame), dtype float/int, shape `(n_obs, >=2)`, no Inf values

Write a validation script if the user's data needs checking. Common issues:
- Normalized counts instead of raw → check `.layers["counts"]`
- Gene symbols in index but no `gene_name` column → copy index to `gene_name`
- Object dtype obs columns → convert to categorical

### Step 2: Run the processing pipeline

```bash
cd src/cellxgene_preprocessing
pixi run snakemake --cores 8 --config 'datasets=["<dataset_name>"]'
```

Key notes:
- GPU accelerates processing (4GB VRAM sufficient). Set `CUDA_VISIBLE_DEVICES` to select GPU.
- Without GPU, increase `--cores` (e.g. 32).
- Memory: allocate ~2x the h5ad file size.
- Cluster captions use GPT-4 API by default (`OPENAI_API_KEY` env var). Without it, falls back to a local Mixtral model (requires 40GB VRAM GPU).
- Output lands in `results/<dataset_name>/`.

### Step 3: Launch cellxgene

```bash
pixi run cellxgene launch -p 5005 --host 0.0.0.0 --max-category-items 500 \
  --var-names gene_name \
  results/<dataset_name>/cellwhisperer_clip_v1/cellxgene.h5ad
```

Access at `http://localhost:5005`. The web app connects to the hosted CellWhisperer API at `cellwhisperer.bocklab.org` for AI features (search, chat).

To self-host the embedding model (4GB VRAM), add:
```bash
--cellwhisperer-clip-model results/models/jointemb/cellwhisperer_clip_v1.ckpt
```

---

## Feature 2: API-Based Cell Scoring

Use the hosted CellWhisperer API to embed text queries and score cells without installing the full model locally. This is useful when an agent or script needs quick cell-type annotations or text-transcriptome similarity scores.

### API endpoints

Base URL: `https://cellwhisperer.bocklab.org/clip/api`

#### Get logit scale (learned CLIP temperature)
```python
import requests
response = requests.get("https://cellwhisperer.bocklab.org/clip/api/logit_scale")
logit_scale = float(response.content)
```

#### Embed text queries
```python
import pickle
import torch
import requests

texts = ["T cell", "B cell", "monocyte"]
response = requests.post(
    "https://cellwhisperer.bocklab.org/clip/api/text_embedding",
    json=texts,
)
text_embeds = torch.from_numpy(pickle.loads(response.content))
# Shape: (len(texts), embedding_dim)
```

#### Score cells against text

Once you have text embeddings and precomputed transcriptome embeddings (from `adata.obsm["transcriptome_embeds"]` in a processed dataset), compute similarity:

```python
import torch

# text_embeds: (n_texts, embedding_dim) from API
# transcriptome_embeds: (n_cells, embedding_dim) from adata.obsm["transcriptome_embeds"]
transcriptome_embeds = torch.from_numpy(adata.obsm["transcriptome_embeds"])

scores = torch.matmul(text_embeds, transcriptome_embeds.t()) * logit_scale
# Shape: (n_texts, n_cells) - higher score = stronger match
```

### Standalone scoring recipe (no CellWhisperer install needed)

For quick annotation of cells that already have precomputed transcriptome embeddings:

```python
import pickle
import requests
import torch
import numpy as np
import anndata

# Load a CellWhisperer-processed dataset
adata = anndata.read_h5ad("results/<dataset>/cellwhisperer_clip_v1/cellxgene.h5ad")
transcriptome_embeds = torch.from_numpy(adata.obsm["transcriptome_embeds"])

# Get model parameters from API
logit_scale = float(requests.get("https://cellwhisperer.bocklab.org/clip/api/logit_scale").content)

# Embed query terms
queries = ["CD8+ cytotoxic T cell", "naive B cell", "classical monocyte"]
response = requests.post("https://cellwhisperer.bocklab.org/clip/api/text_embedding", json=queries)
text_embeds = torch.from_numpy(pickle.loads(response.content))

# Compute per-cell scores
scores = (torch.matmul(text_embeds, transcriptome_embeds.t()) * logit_scale).detach()

# Assign best-matching label per cell
best_labels = [queries[i] for i in scores.argmax(dim=0)]
adata.obs["cellwhisperer_label"] = best_labels
```

---

## Feature 3: Local Library Usage

When explicitly requested, install CellWhisperer as a Python library for local model loading and inference (no API dependency).

### Installation

It uses pixi for dependency management. Infer the user about implications, i.e. that their project would need to be run within pixi, and that pixi would need to be installed (which you could take care of). There is also the option to adapt the environment for `uv` (or `pip`), but this is untested

```bash
# From the cellwhisperer repo root
pixi run pip install -e .
```

Note: this pulls in substantial dependencies (PyTorch, transformers, geneformer). A GPU with >=4GB VRAM is recommended for inference. On CPU, embedding is significantly slower. For quick scoring without local model installation, prefer Feature 2 (API-based scoring).

### Model loading

```python
from cellwhisperer.utils.model_io import load_cellwhisperer_model

# Load from a checkpoint file
pl_model, tokenizer, transcriptome_processor = load_cellwhisperer_model(
    "results/models/jointemb/cellwhisperer_clip_v1.ckpt",
    cache=True,  # enables embedding caching for repeated calls
)
logit_scale = pl_model.model.discriminator.temperature.exp()
```

Model weights can be downloaded from the [project website](http://cellwhisperer.bocklab.org/).

### Embed transcriptomes

```python
import anndata
from cellwhisperer.utils.processing import adata_to_embeds

adata = anndata.read_h5ad("resources/<dataset>/read_count_table.h5ad")

# adata.X must contain raw integer counts
# adata.var must have gene_name column (or gene symbols as index)
transcriptome_embeds = adata_to_embeds(
    adata,
    pl_model.model,
    transcriptome_processor,
    batch_size=32,
)
# Shape: (n_cells, embedding_dim), L2-normalized
```

### Embed texts

```python
text_embeds = pl_model.model.embed_texts(
    ["T cell", "B cell", "monocyte"],
    chunk_size=128,
)
# Shape: (n_texts, embedding_dim), L2-normalized
```

### Score transcriptomes vs texts

```python
from cellwhisperer.utils.inference import score_transcriptomes_vs_texts

scores, group_keys = score_transcriptomes_vs_texts(
    transcriptome_input=transcriptome_embeds,  # or pass adata directly
    text_list_or_text_embeds=text_embeds,       # or pass list of strings
    logit_scale=logit_scale,
    model=pl_model.model,                       # needed if passing raw adata/strings
    transcriptome_processor=transcriptome_processor,  # needed if passing raw adata
    average_mode=None,         # None for per-cell, "embeddings" for per-group average
    score_norm_method=None,    # "zscore", "softmax", "01norm", or None
)
# scores shape: (n_texts, n_cells)
```

### Raw counts validation

```python
from cellwhisperer.utils.processing import ensure_raw_counts_adata

ensure_raw_counts_adata(adata)
# Raises ValueError if neither .X nor .layers["counts"] has integer counts
# If .layers["counts"] has raw counts, it swaps them into .X
```

---

## Troubleshooting

- **`GCC_7.0.0 not found`**: Add `import pyarrow` as the first import in your script.
- **GPU out of memory**: Reduce `batch_size` in `adata_to_embeds` or `score_transcriptomes_vs_texts`.
- **Missing gene_name column**: Copy gene symbols from `.var.index` to `.var["gene_name"]`.
- **Slow processing**: If running with CPU only, increase `--cores` in the snakemake command and expect ~2h per 10k cells on CPU. If GPU is available and , check it's used as intended, and if not suggest to the user to do some environment tests to support this.
