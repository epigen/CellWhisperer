import os
from huggingface_hub import login, hf_hub_download

huggingface_token = snakemake.params.huggingface_token
hoptimus_model_dir = snakemake.params.hoptimus_model_dir

# Login to Hugging Face if token is provided
if huggingface_token:
    login(token=huggingface_token)

# Download the model files
model_repo = "bioptimus/H-optimus-0"

# Download pytorch_model.bin
hf_hub_download(
    repo_id=model_repo,
    filename="pytorch_model.bin",
    local_dir=hoptimus_model_dir,
    force_download=True,
    token=huggingface_token
)

# Download config.json
hf_hub_download(
    repo_id=model_repo,
    filename="config.json",
    local_dir=hoptimus_model_dir,
    force_download=True,
    token=huggingface_token
)

print(f"Downloaded Hoptimus model files to {hoptimus_model_dir}")