#!/bin/bash

conda activate cellwhisperer

pip install "flash-attn<1.0.5" --no-build-isolation

# v0.1.9, but with fix suggested in here: https://github.com/bowang-lab/scGPT/issues/69
# potentially useful: https://github.com/bowang-lab/scGPT/issues/15#issuecomment-1791120487
# or https://github.com/bowang-lab/scGPT/issues/69#issuecomment-1737520314
pip install --no-deps scgpt==0.1.9

echo "Please go to https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y and download the scgpt model files into resources/scGPT_human (as indicated in config.yaml)"
