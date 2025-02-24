#!/bin/bash

mamba env create -f envs/main.yaml  # name: cellwhisperer    this already includes `pip install -e .`
mamba env create -f envs/llava.yaml  # name: llava    this already includes `pip install -e ../modules/LLaVA/[train]`
mamba env create -f envs/llama_cpp.yaml  # name: llama_cpp

# Some package need to be installed after conda -.-
conda activate cellwhisperer

pip install "flash-attn<1.0.5" --no-build-isolation

# v0.1.9, but with fix suggested in here: https://github.com/bowang-lab/scGPT/issues/69
# potentially useful: https://github.com/bowang-lab/scGPT/issues/15#issuecomment-1791120487
# or https://github.com/bowang-lab/scGPT/issues/69#issuecomment-1737520314
pip install --no-deps scgpt==0.1.9

echo "If you want to use scGPT with CellWhisperer, download the model file from https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y and place it into resources/scGPT_human (as indicated in config.yaml)"

conda activate llava

# Install packages that fail to directly install
pip install flash-attn==2.5.3  # required for training
pip install protobuf==3.20.1  # failed 
