#!/bin/bash

HOST=muwhpc1
LLAVA_BASE="Mistral-7B-Instruct-v0.2"
MODEL="cellwhisperer_clip_v1"


# Pull all datasets in config.yaml via rsync
datasets=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('../../config.yaml'))['datasets']))")

for dataset in $datasets; do
    echo $dataset
    mkdir -p /home/moritz/cellwhisperer/results/$dataset/$MODEL
    rsync -avz --partial --progress  $HOST:/msc/home/mschae83/cellwhisperer/results/$dataset/$MODEL/cellxgene.h5ad /home/moritz/cellwhisperer/results/$dataset/$MODEL/cellxgene.h5ad
done

# Also copy the model itself (both LLaVA and cellwhisperer models)
rsync -avz $HOST:/msc/home/mschae83/cellwhisperer/results/models/jointemb/$MODEL.ckpt /home/moritz/cellwhisperer/results/models/jointemb/$MODEL.ckpt

rsync -avz $HOST:/msc/home/mschae83/cellwhisperer/results/llava/finetuned/${LLAVA_BASE}__${MODEL} /home/moritz/cellwhisperer/results/llava/finetuned/
