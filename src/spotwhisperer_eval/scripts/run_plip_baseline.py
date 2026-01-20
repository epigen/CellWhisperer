#!/usr/bin/env python3
# Minimal wrapper to run PLIP baseline exactly like external script
import sys
import os
import numpy as np
import pandas as pd
import torch
import anndata
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Terms
TERMS_1 = ['Background', 'B cells', 'Macrophages/Monocytes', 'Adipocytes', 'Dendritic cells', 'T cells', 'Granulocytes', 'NK cells', 'Nerves', 'Plasma cells', 'Smooth muscle', 'Stroma', 'Tumor cells', 'Vasculature/Lymphatics', 'Other cells']
TERMS_2 = ['A sample of Background cells', 'A sample of B cells', 'A sample of Macrophages/Monocytes cells', 'A sample of Adipocytes cells', 'A sample of Dendritic cells', 'A sample of T cells', 'A sample of Granulocytes cells', 'A sample of NK cells', 'A sample of Nerves cells', 'A sample of Plasma cells', 'A sample of Smooth muscle cells', 'A sample of Stroma cells', 'A sample of Tumor cells', 'A sample of Vasculature/Lymphatics cells', 'A sample of Other cells']

# Paths: use processed PathoCell files from project structure
data_dir = Path(snakemake.params.data_dir)
out_terms1 = Path(snakemake.output.logits_terms1)
out_terms2 = Path(snakemake.output.logits_terms2)

# Collect files
tiff_files = sorted([p for p in data_dir.glob('*.tiff')])
h5ad_files = sorted([p for p in data_dir.glob('*.h5ad')])
assert len(tiff_files) == len(h5ad_files)

labels_1 = TERMS_1
labels_2 = TERMS_2

# Initialize empty DFs
df_terms1 = pd.DataFrame(columns=['source_image', 'spot_id'] + [f'{label}' for label in labels_1])
df_terms2 = pd.DataFrame(columns=['source_image', 'spot_id'] + [f'{label}' for label in labels_2])

# Device/model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CLIPModel.from_pretrained('vinid/plip').to(device=device)
processor = CLIPProcessor.from_pretrained('vinid/plip')

for tiff_path, h5ad_path in zip(tiff_files, h5ad_files):
    source_image = tiff_path.name
    full_image = Image.open(tiff_path).convert('RGB')
    img_w, img_h = full_image.size

    adata = anndata.read_h5ad(h5ad_path)
    adata.obs['x_pixel'] = adata.obs['x_pixel'].astype(int)
    adata.obs['y_pixel'] = adata.obs['y_pixel'].astype(int)

    crop_diameter_pixels = 224
    half_d = crop_diameter_pixels // 2

    results_terms1 = []
    results_terms2 = []

    for i in range(len(adata)):
        spot_id = adata.obs.index[i]
        x_pixel = int(adata.obs['x_pixel'].iloc[i])
        y_pixel = int(adata.obs['y_pixel'].iloc[i])

        x_start = max(0, x_pixel - half_d)
        y_start = max(0, y_pixel - half_d)
        x_end = min(img_w, x_pixel + half_d)
        y_end = min(img_h, y_pixel + half_d)

        crop = np.array(full_image.crop((x_start, y_start, x_end, y_end)))
        actual_h, actual_w = crop.shape[:2]
        pad_y = crop_diameter_pixels - actual_h
        pad_x = crop_diameter_pixels - actual_w
        if pad_y > 0 or pad_x > 0:
            crop = np.pad(crop, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant', constant_values=0)
        crop = crop[:, :, :3]

        # Inference for TERMS_1
        inputs1 = processor(text=TERMS_1, images=crop, return_tensors='pt', padding=True).to(device=device)
        outputs1 = model(**inputs1)
        logits1 = outputs1.logits_per_image[0].detach().cpu().numpy()

        # Inference for TERMS_2
        inputs2 = processor(text=TERMS_2, images=crop, return_tensors='pt', padding=True).to(device=device)
        outputs2 = model(**inputs2)
        logits2 = outputs2.logits_per_image[0].detach().cpu().numpy()

        row_base = {'source_image': source_image, 'spot_id': spot_id}
        row_terms1 = {**row_base, **{labels_1[j]: logits1[j] for j in range(len(logits1))}}
        row_terms2 = {**row_base, **{labels_2[j]: logits2[j] for j in range(len(logits2))}}
        results_terms1.append(row_terms1)
        results_terms2.append(row_terms2)

    df_terms1 = pd.concat([df_terms1, pd.DataFrame(results_terms1)], ignore_index=True)
    df_terms2 = pd.concat([df_terms2, pd.DataFrame(results_terms2)], ignore_index=True)

out_terms1.parent.mkdir(parents=True, exist_ok=True)
out_terms2.parent.mkdir(parents=True, exist_ok=True)
df_terms1.to_csv(out_terms1, index=False)
df_terms2.to_csv(out_terms2, index=False)
