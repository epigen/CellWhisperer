# Deprecated UCE Configurations

**Date:** March 3, 2026

## Notice

The following configuration files in this experiment reference **deprecated UCE models**:

- `delta_config/uce4_frozen.yaml` - References old 4-layer UCE model
- `delta_config/uce4_finetune.yaml` - References old 4-layer UCE model  
- `delta_config/uce33_frozen.yaml` - References old 33-layer UCE model

## Migration

**Old UCE implementation** (deprecated):
- Models: `uce4`, `uce33` (from epigen/modules/UCE fork)
- Model files: `resources/UCE/4layer_model.torch`, `resources/UCE/33layer_model.torch`
- Required protein embeddings, species offsets, chromosome data

**New UCE implementation** (current):
- Model: `uce` (Kuan's implementation from data_collection_exp)
- Checkpoint: `KuanP/uce-cosmx-geneset` (auto-downloaded from HuggingFace)
- Gene data: `static/UCE/gene_names.txt`, `static/UCE/all_species_gene_dict.json`
- Embedding dimension: 512 (vs old 1280)

## How to Update Configs

To use the new UCE model, replace references:
- Change `uce4` or `uce33` → `uce`
- Update best_model_path if needed
- The new model uses a cleaner API with simplified gene tokenization

## Background

The old UCE models required complex preprocessing (protein embeddings, chromosome mappings, species offsets) and had compatibility issues with modern dependencies. The new implementation from Kuan uses a transformer-based architecture with simplified tokenization and is maintained actively.

For details, see:
- New implementation: `modules/data_collection_exp/` (git submodule)
- Model wrapper: `src/cellwhisperer/jointemb/uce_model.py`
- Config: `config.yaml` (section `uce_paths`)
