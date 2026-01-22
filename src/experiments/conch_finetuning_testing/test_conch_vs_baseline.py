import os
import glob
import math
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import numpy as np
import pandas as pd
import anndata
from PIL import Image

# CONCH imports
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer


TERMS_1 = [
    "Background",
    "B cells",
    "Macrophages/Monocytes",
    "Adipocytes",
    "Dendritic cells",
    "T cells",
    "Granulocytes",
    "NK cells",
    "Nerves",
    "Plasma cells",
    "Smooth muscle",
    "Stroma",
    "Tumor cells",
    "Vasculature/Lymphatics",
    "Other cells",
]


def _crop_center(image: Image.Image, x: int, y: int, d: int) -> Image.Image:
    """Crop a d x d tile centered at (x, y); pad with zeros if near edges."""
    w, h = image.size
    half = d // 2
    x0 = max(0, x - half)
    y0 = max(0, y - half)
    x1 = min(w, x + half)
    y1 = min(h, y + half)
    crop = np.array(image.crop((x0, y0, x1, y1)))
    # Pad if smaller
    pad_y = d - crop.shape[0]
    pad_x = d - crop.shape[1]
    if pad_y > 0 or pad_x > 0:
        crop = np.pad(
            crop, ((0, pad_y), (0, pad_x), (0, 0)), mode="constant", constant_values=0
        )
    crop = crop[:, :, :3]
    return Image.fromarray(crop)


def load_image_from_adata(adata: anndata.AnnData) -> Image.Image:
    """Load the underlying slide image referenced in AnnData.uns."""
    if "image_path" in adata.uns:
        return Image.open(adata.uns["image_path"]).convert("RGB")
    if "he_slide" in adata.uns and isinstance(adata.uns["he_slide"], Image.Image):
        return adata.uns["he_slide"].convert("RGB")
    if "20x_slide" in adata.uns and isinstance(adata.uns["20x_slide"], Image.Image):
        return adata.uns["20x_slide"].convert("RGB")
    # If an OpenSlide object is present but no path, this simple loader won't handle it.
    # Prefer using image_path for this test to match baseline cropping.
    raise ValueError(
        "AnnData.uns does not contain a supported image reference (image_path, he_slide, 20x_slide)"
    )


def compute_conch_scores_for_h5ad(
    h5ad_path: Path, hf_token: Optional[str] = None
) -> pd.DataFrame:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        "hf_hub:MahmoodLab/conch",
        hf_auth_token=hf_token,
    )
    model.to(device)

    tokenizer = get_tokenizer()
    tokenized_prompts = tokenize(texts=TERMS_1, tokenizer=tokenizer).to(device)

    adata = anndata.read_h5ad(str(h5ad_path))
    # Ensure integer pixel coords
    x_pix = adata.obs["x_pixel"].astype(int)
    y_pix = adata.obs["y_pixel"].astype(int)

    image = load_image_from_adata(adata)
    src_name = Path(adata.uns.get("image_path", "unknown.tiff")).name

    rows = []
    for spot_id, x, y in zip(adata.obs.index.tolist(), x_pix.tolist(), y_pix.tolist()):
        tile = _crop_center(image, x, y, 224)
        inp = preprocess(tile).unsqueeze(0).to(device)
        with torch.inference_mode():
            img_feat = model.encode_image(inp)  # normalized features
            txt_feat = model.encode_text(tokenized_prompts)
            sims = (img_feat @ txt_feat.T * model.logit_scale.exp()).cpu().numpy()[0]
        row = {"source_image": src_name, "spot_id": spot_id}
        for j, lbl in enumerate(TERMS_1):
            row[lbl] = float(sims[j])
        rows.append(row)

    return pd.DataFrame(rows)


def correlate_with_baseline(df_new: pd.DataFrame, baseline_csv: Path) -> pd.DataFrame:
    base = pd.read_csv(baseline_csv)
    # Align on spot_id (and source_image if present in both)
    join_cols = ["spot_id"]
    if "source_image" in base.columns and "source_image" in df_new.columns:
        join_cols.append("source_image")
    merged = df_new.merge(base, on=join_cols, suffixes=("_new", "_base"))
    if merged.empty:
        raise ValueError(
            "No overlapping rows between computed scores and baseline CSV; check source_image/spot_id alignment."
        )

    corrs = []
    for lbl in TERMS_1:
        x = merged[f"{lbl}_new"].to_numpy()
        y = merged[f"{lbl}"] if f"{lbl}" in merged.columns else merged[f"{lbl}_base"]
        y = y.to_numpy()
        if x.size == 0 or y.size == 0:
            r = np.nan
        else:
            r = np.corrcoef(x, y)[0, 1]
        corrs.append({"label": lbl, "pearson_r": r})
    return pd.DataFrame(corrs)


def main():
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    # Use local file in the same directory as the script
    script_dir = Path(__file__).resolve().parent
    matches = sorted(script_dir.glob("reg001_A_patch*.h5ad"))
    if not matches:
        raise FileNotFoundError(
            f"No reg001_A_patch*.h5ad found in {script_dir}. Please place the file next to the script."
        )
    h5ad_path = matches[0]

    df_new = compute_conch_scores_for_h5ad(h5ad_path, hf_token=hf_token)

    baseline_csv = Path(
        "/home/moritz/Projects/SpatialWhisperer/plip_conch_baseline_performance/conch_logits_terms1.csv"
    )
    corrs = correlate_with_baseline(df_new, baseline_csv)

    print("Correlation with baseline (TERMS_1):")
    print(corrs.to_string(index=False))
    # Optional sanity: warn if any correlation is far from 1
    low = corrs["pearson_r"].dropna() < 0.95
    if low.any():
        bad = corrs.loc[low]
        print(
            "Warning: Some labels have correlation < 0.95:\n",
            bad.to_string(index=False),
        )


if __name__ == "__main__":
    main()
