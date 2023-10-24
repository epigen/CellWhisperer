# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import json

# Load data from snakemake.input
with open(snakemake.input[0], "r") as f:
    data = json.load(f)

# Extract sentences from data (dict)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
model = AutoModel.from_pretrained("microsoft/biogpt")
model.eval()
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = model.to(device)
# Tokenize sentences

encoded_input = tokenizer(
    list(data.values()),
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

# Compute token embeddings
model_output = model(**encoded_input.to(device))

# Perform pooling. In this case, mean pooling
sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

# Use numpy to save embeddings to snakemake.output (npz file). Make sure to save the keys with it
np.savez_compressed(
    snakemake.output[0],
    ids=list(data.keys()),
    embeddings=sentence_embeddings.detach().cpu().numpy(),
)
