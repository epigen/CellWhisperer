#!/bin/bash

# Install conda environments
conda env create -f envs/main.yaml  # name: cellwhisperer; includes pip installation of submodules
conda env create -f envs/llava.yaml  # name: llava; includes pip installation of ../modules/LLaVA/[train]`
conda env create -f envs/llama_cpp.yaml  # name: llama_cpp

# Build cellxgene web app (for details, refer to /modules/cellxgene/dev_docs/developer_guidelines.md)
conda activate cellwhisperer
source activate cellwhisperer  # docker compatibility

cd modules/cellxgene
make build-for-server-dev

# Install scGPT
# pip install "flash-attn<1.0.5" --no-build-isolation  # required for scgpt, but causes issues in some environments
pip install --no-deps scgpt==0.1.9

# Install remaining LLaVA packages
conda activate llava
source activate llava  # docker compatibility

pip install flash-attn==2.5.3  # required for training only
pip install protobuf==3.20.1  # failed to install within pyproject.toml requirements

echo "If you want to use scGPT with CellWhisperer, download the model file from https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y and place it into resources/scGPT_human (as indicated in config.yaml)"
