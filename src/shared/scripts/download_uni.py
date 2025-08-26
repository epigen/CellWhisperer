import os
from huggingface_hub import login, hf_hub_download
import shutil

huggingface_token = snakemake.params.huggingface_token

# Login to Hugging Face if token is provided
if huggingface_token:
    login(token=huggingface_token)

# Download the model files

# Download pytorch_model.bin
model_path = hf_hub_download(
    repo_id=snakemake.params.model_name,
    filename="pytorch_model.bin",
    force_download=True,
    token=huggingface_token,
)
shutil.copy2(model_path, snakemake.output.model_path)


# Download config.json
config_path = hf_hub_download(
    repo_id=snakemake.params.model_name,
    filename="config.json",
    force_download=True,
    token=huggingface_token,
)

shutil.copy2(config_path, snakemake.output.config_path)
