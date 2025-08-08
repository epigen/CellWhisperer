import os
from huggingface_hub import login, hf_hub_download

huggingface_token = snakemake.params.huggingface_token
uni_model_dir = snakemake.params.uni_model_dir

# Login to Hugging Face if token is provided
if huggingface_token:
    login(token=huggingface_token)

# Download the model files

# Download pytorch_model.bin
hf_hub_download(
    repo_id=snakemake.params.model_name,
    filename="pytorch_model.bin",
    local_dir=uni_model_dir,
    force_download=True,
    token=huggingface_token,
)


    repo_id=snakemake.params.model_name,
    filename="config.json",
    local_dir=uni_model_dir,
    force_download=True,
    token=huggingface_token,
)