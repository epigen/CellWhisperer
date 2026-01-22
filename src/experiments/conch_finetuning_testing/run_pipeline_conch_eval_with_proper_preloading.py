import os
import json
from pathlib import Path
from typing import List

import torch
import numpy as np
import pandas as pd
import anndata

from cellwhisperer.config import model_path_from_name
from cellwhisperer.validation.zero_shot.functions import (
    get_performance_metrics_left_vs_right,
)
from cellwhisperer.jointemb.config import TranscriptomeTextDualEncoderConfig
from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel
from cellwhisperer.misc.debug import start_debugger
from conch.open_clip_custom import get_tokenizer, tokenize
from conch.open_clip_custom.transform import image_transform
from PIL import Image


def build_cell_type_prompts(class_names: List[str]) -> List[str]:
    return [
        f"A sample of {ct}" if ct.endswith("cells") else f"A sample of {ct} cells"
        for ct in class_names
    ]


def load_model():
    # Initialize a fresh model from config (CONCH image + identity projection for eval parity)
    cfg = TranscriptomeTextDualEncoderConfig(
        transcriptome_model_type="geneformer",
        text_model_type="conch_text",
        text_config={"model_type": "conch_text"},
        image_model_type="conch_image",
        image_config={
            "model_type": "conch_image",
            "model_cfg": "conch_ViT-B-16",
            "checkpoint_path": "hf_hub:MahmoodLab/conch",
        },
        identity_projection=True,
        projection_dim=512,
    )
    model = TranscriptomeTextDualEncoderModel(config=cfg)

    transcriptome_model_directory = model_path_from_name(
        model.model.transcriptome_model.config.model_type
    )
    # if text_model_name_or_path is None:
    text_model_name_or_path = model_path_from_name(
        model.model.text_model.config.model_type
    )

    image_model_name_or_path = model_path_from_name(
        model.model.image_model.config.model_type
    )

    kwargs = model.config.to_dict()
    kwargs["transcriptome_model"] = "geneformer"
    kwargs["text_model"] = "conch_text"
    kwargs["image_model"] = "conch_image"

    TranscriptomeTextDualEncoderModel.from_transcriptome_text_pretrained(
        transcriptome_model_name_or_path=transcriptome_model_directory,
        text_model_name_or_path=text_model_name_or_path,
        image_model_name_or_path=image_model_name_or_path,
        **kwargs,
    )


def main():
    # start_debugger(wait_for_client=True, port=5685)
    # Inputs via env for simplicity
    script_dir = Path(__file__).resolve().parent
    h5ad_path = script_dir / "reg001_A_patch.h5ad"
    prediction_level = os.environ.get("PREDICTION_LEVEL", "patch")  # or "cell"

    assert h5ad_path and h5ad_path.exists(), "Set H5AD to an existing .h5ad path"

    # Load data
    adata = anndata.read_h5ad(str(h5ad_path))

    # Use fixed pathology TERMS_1 to match baseline
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
    prompts = TERMS_1

    # First-row, baseline-identical scoring using CONCH preprocess + encode
    def _load_image_from_adata(adata: anndata.AnnData) -> Image.Image:
        if "image_path" in adata.uns:
            return Image.open(adata.uns["image_path"]).convert("RGB")
        if "he_slide" in adata.uns and isinstance(adata.uns["he_slide"], Image.Image):
            return adata.uns["he_slide"].convert("RGB")
        if "20x_slide" in adata.uns and isinstance(adata.uns["20x_slide"], Image.Image):
            return adata.uns["20x_slide"].convert("RGB")
        raise ValueError(
            "AnnData.uns does not contain a supported image reference (image_path, he_slide, 20x_slide)"
        )

    def _crop_center(image: Image.Image, x: int, y: int, d: int) -> Image.Image:
        w, h = image.size
        half = d // 2
        x0 = max(0, x - half)
        y0 = max(0, y - half)
        x1 = min(w, x + half)
        y1 = min(h, y + half)
        crop = np.array(image.crop((x0, y0, x1, y1)))
        pad_y = d - crop.shape[0]
        pad_x = d - crop.shape[1]
        if pad_y > 0 or pad_x > 0:
            crop = np.pad(
                crop,
                ((0, pad_y), (0, pad_x), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        crop = crop[:, :, :3]
        return Image.fromarray(crop)

    # unwrap to raw CoCa model
    conch_model = model.image_model
    if hasattr(conch_model, "model"):
        conch_model = conch_model.model
    if hasattr(conch_model, "model"):
        conch_model = conch_model.model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = _load_image_from_adata(adata)
    x = int(adata.obs["x_pixel"].iloc[0])
    y = int(adata.obs["y_pixel"].iloc[0])
    tile = _crop_center(image, x, y, 224)
    preprocess = image_transform(image_size=448)
    inp = preprocess(tile).unsqueeze(0).to(device)

    tokenizer = get_tokenizer()
    tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer).to(device)

    conch_model.to(device)
    with torch.inference_mode():
        img_feat = conch_model.encode_image(inp)
        txt_feat = conch_model.encode_text(tokenized_prompts)
        sims = (img_feat @ txt_feat.T * conch_model.logit_scale.exp()).cpu().numpy()[0]

    print("Scores (first row) against TERMS_1):")
    for lbl, val in zip(prompts, sims.tolist()):
        print(f"{lbl}: {val:.6f}")


if __name__ == "__main__":
    main()
