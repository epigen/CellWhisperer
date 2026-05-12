---
name: cellwhisperer
description: |
  Use when the user needs to work with single-cell RNA-seq data using CellWhisperer: scoring/annotating cells with free-text queries (cell types, states, pathways), launching the interactive cellxgene browser, or any request involving transcriptome-to-text similarity. Triggers on: "cellwhisperer", "cell type annotation", "scRNA-seq analysis", "single-cell scoring", "cellxgene", "stemness", or questions about cell identity/state in transcriptomic data.
---

# CellWhisperer

CellWhisperer is a multimodal AI model that scores single cells against free-text queries (cell types, states, pathways) using a CLIP-style joint embedding of transcriptomes and natural language. Published in [Nature Biotechnology](https://doi.org/10.1038/s41587-025-02857-9).

## Plugin setup

This skill is distributed as a Claude Code plugin. Installing it clones the full CellWhisperer repository (with pixi environment) to `~/.claude/plugins/cache/cellwhisperer/cellwhisperer/<version>/`. This clone is used for all compute — no separate installation needed.

```bash
# generally prevent auto-update for your safety
claude plugin marketplace add epigen/cellwhisperer@v0.1.0
claude plugin install cellwhisperer@cellwhisperer
```

The clone path (referred to as `$CW_ROOT` below) is:
```
~/.claude/plugins/cache/cellwhisperer/cellwhisperer/0.1.0/
```

Before starting, read `$CW_ROOT/README.md` for full project context.

---

## Workflow 1: Score & Analyze (default)

Score cells in any h5ad dataset against free-text queries. This is the primary workflow.

### Step 1: Compute transcriptome embeddings

Most h5ad files won't have CellWhisperer embeddings. Compute them using the plugin's pixi environment and model checkpoint:

```bash
cd ~/.claude/plugins/cache/cellwhisperer/cellwhisperer/0.1.0/
pixi run python /path/to/embed_script.py
```

The embedding script should:

```python
import anndata
import torch
from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.utils.processing import adata_to_embeds, ensure_raw_counts_adata

# Load model (ships with the plugin)
pl_model, tokenizer, transcriptome_processor = load_cellwhisperer_model(
    "results/models/jointemb/cellwhisperer_clip_v1.ckpt",
    cache=True,
)

# Load user's dataset
adata = anndata.read_h5ad("/absolute/path/to/user_data.h5ad")

# Validate raw counts (required)
ensure_raw_counts_adata(adata)

# Compute embeddings
transcriptome_embeds = adata_to_embeds(
    adata,
    pl_model.model,
    transcriptome_processor,
    batch_size=32,  # reduce if GPU OOM
)

# Save back
adata.obsm["transcriptome_embeds"] = transcriptome_embeds.cpu().numpy()
adata.write_h5ad("/absolute/path/to/user_data.h5ad")
```

**h5ad requirements** for embedding:
- Raw integer counts in `.X` or `.layers["counts"]` (int32, no NaN)
- `.var` must have a `gene_name` column with gene symbols
- Recommended: `ensembl_id` in `.var` (computed if missing)

**Performance**: GPU (>=4GB VRAM) recommended. CPU works but is significantly slower.

If the h5ad already has `transcriptome_embeds` in `.obsm`, skip this step.

### Step 2: Score cells via the API

Once embeddings exist, score cells against any text query using the hosted API. This step requires only `requests`, `pickle`, and `torch` — no CellWhisperer install needed in the user's environment.

```python
import pickle
import requests
import torch
import anndata

# Load dataset with embeddings
adata = anndata.read_h5ad("/path/to/data.h5ad")
transcriptome_embeds = torch.from_numpy(adata.obsm["transcriptome_embeds"])

# Get logit scale from API
logit_scale = float(requests.get(
    "https://cellwhisperer.bocklab.org/clip/api/logit_scale"
).content)

# Embed text queries via API
queries = ["intestinal stem cell", "inflamed cell", "goblet cell"]
response = requests.post(
    "https://cellwhisperer.bocklab.org/clip/api/text_embedding",
    json=queries,
)
text_embeds = torch.from_numpy(pickle.loads(response.content))

# Score: (n_queries, n_cells), higher = stronger match
scores = (torch.matmul(text_embeds, transcriptome_embeds.t()) * logit_scale).detach()

# Add scores to adata
for i, q in enumerate(queries):
    adata.obs[f"cw_score_{q}"] = scores[i].numpy()
```

### Step 3: Analyze and plot

Use the scores for downstream analysis — violin plots by condition, UMAP overlays, statistical tests, cluster-level summaries, etc. Write analysis code in the user's working directory (not in `$CW_ROOT`).

### Alternative: score locally without API

For offline use or to avoid API dependency, score text queries locally using the plugin's pixi environment (run from `$CW_ROOT`):

```python
# Text embedding (local, no API)
text_embeds = pl_model.model.embed_texts(
    ["intestinal stem cell", "inflamed cell"],
    chunk_size=128,
)

# Score
scores = (torch.matmul(text_embeds, transcriptome_embeds.t())
          * pl_model.model.discriminator.temperature.exp()).detach()
```

---

## Workflow 2: Interactive Browser

Process a raw h5ad through the full CellWhisperer pipeline and launch the cellxgene web app with AI-powered search and chat. Run all commands from `$CW_ROOT`.

### Step 1: Prepare the dataset

Place the h5ad at `$CW_ROOT/resources/<dataset_name>/read_count_table.h5ad`.

Requirements are the same as Workflow 1 (raw counts, gene_name column). Additionally:
- Use `categorical` dtype for categorical `.obs` columns
- 2D embeddings in `.obsm` must be `np.ndarray` (not DataFrame), no Inf values

### Step 2: Run the preprocessing pipeline

```bash
cd $CW_ROOT/src/cellxgene_preprocessing
pixi run snakemake --cores 8 --config 'datasets=["<dataset_name>"]'
```

This computes embeddings, generates cluster labels (via GPT-4 API if `OPENAI_API_KEY` is set, otherwise local Mixtral), and prepares the cellxgene-ready h5ad.

Output: `$CW_ROOT/results/<dataset_name>/cellwhisperer_clip_v1/cellxgene.h5ad`

### Step 3: Launch cellxgene

```bash
pixi run cellxgene launch -p 5005 --host 0.0.0.0 --max-category-items 500 \
  --var-names gene_name \
  results/<dataset_name>/cellwhisperer_clip_v1/cellxgene.h5ad
```

Access at `http://localhost:5005`. The web app uses the hosted API for AI features (search, chat).

---

## Troubleshooting

- **`GCC_7.0.0 not found`**: Add `import pyarrow` as the first import in your script.
- **GPU out of memory**: Reduce `batch_size` in `adata_to_embeds`.
- **Missing gene_name column**: Copy gene symbols from `.var.index` to `.var["gene_name"]`.
- **Non-integer counts**: `ensure_raw_counts_adata(adata)` detects and fixes this (swaps `.layers["counts"]` into `.X` if needed).
- **Slow on CPU**: Embedding is significantly slower without GPU. Consider reducing the dataset size or using a machine with GPU access.
