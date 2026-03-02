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

print("Imports and setup complete.")

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
model = CLIPModel.from_pretrained("vinid/plip").to(device=device)
processor = CLIPProcessor.from_pretrained("vinid/plip")

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
    inputs1 = processor(text=TERMS_1, images=full_image, return_tensors="pt", padding=True).to(device=device)
    outputs1 = model(**inputs1)
    logits1 = outputs1.logits_per_image[0].detach().cpu().numpy()

    # Inference for TERMS_2 (separate pass)
    inputs2 = processor(text=TERMS_2, images=full_image, return_tensors="pt", padding=True).to(device=device)
    outputs2 = model(**inputs2)
    logits2 = outputs2.logits_per_image[0].detach().cpu().numpy()

    # Append to results (dict for easy DF conversion)
    row_base = {'source_image': source_image, 'spot_id': barcode}
    row_terms1 = {**row_base, **{f'{labels_1[j]}': logits1[j] for j in range(len(logits1))}}
    row_terms2 = {**row_base, **{f'{labels_2[j]}': logits2[j] for j in range(len(logits2))}}
    results_terms1.append(row_terms1)
    results_terms2.append(row_terms2)

    # Append this file's results to main DFs
    df_terms1 = pd.concat([df_terms1, pd.DataFrame(results_terms1)], ignore_index=True)
    df_terms2 = pd.concat([df_terms2, pd.DataFrame(results_terms2)], ignore_index=True)

# Save aggregated DFs
df_terms1.to_csv("plip_logits_terms1_lc.csv", index=False)
df_terms2.to_csv("plip_logits_terms2_lc.csv", index=False)
print("Aggregated results saved.")