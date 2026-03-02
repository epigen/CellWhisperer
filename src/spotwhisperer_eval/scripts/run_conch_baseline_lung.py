# start coding here
import os
import json
import numpy as np
import pandas as pd
import torch
import anndata
from tqdm import tqdm
from pathlib import Path
import glob
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# CONCH-specific imports (adjust as needed)
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer

# Suppress warnings and logging if needed (add based on your env)
import warnings
import logging
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger().setLevel(logging.ERROR)

print("Imports and setup complete.")

# Get hf token from environment variable
hf_token = os.getenv('HUGGINGFACE_TOKEN')

# Define the terms for classification
TERMS_1 = ["tumor cells", "normal cells", "tertiary lymphoid structure", "infiltrating cells"]
TERMS_2 = ["epithelial cells", "endothelial cells", "immune cells", "stromal cells"]


tiles_dir = Path("/msc/home/aawast98/cellspot/VLM_benchmarks/data/tiles/sw_size")
image_files = []
for i in range (1,6):
    image_path = os.path.join(tiles_dir, f"lc_{i}", "*.jpg")
    image_files += glob.glob(image_path)

labels_1 = TERMS_1
labels_2 = TERMS_2

# Initialize empty DFs
df_terms1 = pd.DataFrame(columns=['source_image', 'spot_id'] + [f'{label}' for label in labels_1])
df_terms2 = pd.DataFrame(columns=['source_image', 'spot_id'] + [f'{label}' for label in labels_2])

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load CONCH model and preprocessor
model, preprocess = create_model_from_pretrained('conch_ViT-B-16',
                                                    "hf_hub:MahmoodLab/conch",
                                                    hf_auth_token=hf_token)
model.to(device)

tokenizer = get_tokenizer()
tokenized_prompts_1 = tokenize(texts=TERMS_1, tokenizer=tokenizer).to(device)
tokenized_prompts_2 = tokenize(texts=TERMS_2, tokenizer=tokenizer).to(device)

for image_path in tqdm(image_files):
    # source_image is the folder in which images are stored
    # barcode is the name of the image without the extension
    source_image = image_path.split("/")[-2]
    barcode = Path(image_path).stem

    # Loading the image sometime fails, so we need to handle it
    try:
        full_image = Image.open(image_path).convert("RGB")
        # Check if image loaded successfully
        if full_image.size == (0, 0):
            raise ValueError(f"Image {image_path} loaded with zero size.")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        continue  # Skip this file if loading fails

    # Collect results for this file
    results_terms1 = []
    results_terms2 = []
    
        
    # Inference for TERMS_1
    inputs = preprocess(full_image).unsqueeze(0).to(device=device)
    # Run inference
    with torch.inference_mode():
        image_embeddings = model.encode_image(inputs)
        text_embeddings = model.encode_text(tokenized_prompts_1)
        sim_scores_1 = (image_embeddings @ text_embeddings.T * model.logit_scale.exp()).cpu().numpy()[0]

    # Inference for TERMS_2 (separate pass)

    with torch.inference_mode():
        image_embeddings_2 = model.encode_image(inputs)
        text_embeddings_2 = model.encode_text(tokenized_prompts_2)
        sim_scores_2 = (image_embeddings_2 @ text_embeddings_2.T * model.logit_scale.exp()).cpu().numpy()[0]


    # Append to results (dict for easy DF conversion)
    row_base = {'source_image': source_image, 'spot_id': barcode}
    row_terms1 = {**row_base, **{f'{labels_1[j]}': sim_scores_1[j] for j in range(len(sim_scores_1))}}
    row_terms2 = {**row_base, **{f'{labels_2[j]}': sim_scores_2[j] for j in range(len(sim_scores_2))}}
    results_terms1.append(row_terms1)
    results_terms2.append(row_terms2)

    # Append this file's results to main DFs
    df_terms1 = pd.concat([df_terms1, pd.DataFrame(results_terms1)], ignore_index=True)
    df_terms2 = pd.concat([df_terms2, pd.DataFrame(results_terms2)], ignore_index=True)

# Save aggregated DFs
df_terms1.to_csv("conch_logits_terms1_lc.csv", index=False)
df_terms2.to_csv("conch_logits_terms2_lc.csv", index=False)
print("Aggregated results saved.")
