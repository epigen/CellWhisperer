#!/bin/bash

# Install conda environments
conda env create -f envs/main.yaml  # name: cellwhisperer; includes pip installation of submodules
conda env create -f envs/llava.yaml  # name: llava; includes pip installation of ../modules/LLaVA/[train]`
conda env create -f envs/llama_cpp.yaml  # name: llama_cpp
conda env create -f envs/hest.yaml  # name: hest

# Build cellxgene web app (for details, refer to /modules/cellxgene/dev_docs/developer_guidelines.md)
conda activate cellwhisperer
source activate cellwhisperer  # docker compatibility

cd modules/cellxgene
make build-for-server-dev

# Install scGPT
pip install "flash-attn<1.0.5" --no-build-isolation
pip install --no-deps scgpt==0.1.9

# Install remaining LLaVA packages
conda activate llava
source activate llava  # docker compatibility

pip install flash-attn==2.5.3  # required for training only
pip install protobuf==3.20.1  # failed to install within pyproject.toml requirements

conda activate hest  # docker compatibility
source activate hest
pip install --no-deps git+https://github.com/mahmoodlab/hest.git@6759f3d25932f17d6cd56d7bfd4d3651376e76f9
