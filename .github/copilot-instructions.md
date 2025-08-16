# CellWhisperer Development Instructions

Always follow these instructions first and only fall back to additional search and context gathering if the information here is incomplete or found to be in error.

CellWhisperer and SpotWhisperer are multimodal AI models for single-cell RNA sequencing and histopathology analysis that combine transcriptomics, H&E-images and natural language processing. They include both embedding and chat models, with a web application based on CELLxGENE Explorer.

## Critical Build Timing Expectations

Probably all too long to run within github servers

- **Environment Setup**: 60-90 minutes
- **CELLxGENE Build**: 30-45 minutes  
- **Model Tests**: 30-45 minutes
- **Full Pipeline**: 2-3 days
- **Dataset Processing**: 4-24 hours per dataset

**NEVER CANCEL** any of these long-running operations. Always set appropriate timeouts with 50%+ buffer.

## Key Directories and Components

### Source Code Structure
- `src/cellwhisperer/`: Core Python package (embedding model, training, inference)
- `src/figures/`: Pipeline for manuscript plots and analyses
- `src/cellxgene_preprocessing/`: Dataset preprocessing for web app
- `src/datasets/`: Training/validation dataset preparation
- `src/llava/`: Chat model training pipeline
- `src/test/`: Model testing (limited test coverage)

### Configuration and Build
- `config.yaml`: Main configuration file with model paths and dataset definitions  
- `envs/`: Conda environment definitions
- `pyproject.toml`: Python package definition
- `Dockerfile`: Complete containerized build

### Submodules (in modules/)
- `cellxgene`: Modified CELLxGENE Explorer with CellWhisperer integration
- `LLaVA`: Chat model components  
- `Geneformer`: Transcriptome model
- `UCE`, `DeepSpot`: Additional model components

## Development Workflow

1. **Always start with environment setup**: `./envs/setup.sh`
2. **For code changes**: Work in `cellwhisperer` conda environment
3. **For pipeline changes**: Test with `snakemake --dry-run` first
5. **Always validate**: Implement and run model tests and pipeline checks before committing

## Resource Requirements

- **Minimum**: 16GB RAM, 4GB VRAM GPU for basic functionality
- **Recommended**: 64GB+ RAM, 40GB VRAM GPU for full capabilities  
- **Full Pipeline**: 1TB+ storage, 1TB RAM, 40GB VRAM, multi-day runtime
- **Network**: Stable high-bandwidth connection for downloading models and datasets

This codebase requires substantial computational resources and long build times. Plan accordingly and never cancel long-running operations.
