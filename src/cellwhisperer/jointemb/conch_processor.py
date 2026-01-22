import logging
import os
from typing import Any, Dict, Optional, Union

import anndata
import numpy as np
import torch
from PIL import Image
from transformers.processing_utils import ProcessorMixin

from conch.open_clip_custom.transform import image_transform


logger = logging.getLogger(__name__)


class ConchProcessor(ProcessorMixin):
    attributes = []

    def __init__(self, *args, **kwargs):
        # Match baseline crop size for spot-centered tiles
        self.context_diameter = 224
        self.cell_diameter = 56
        # Use CONCH's standard transform (Resize + CenterCrop + Normalize); default size 448
        self.transform_ctx = image_transform(image_size=448)
        # Cell view is not used by CONCH encoder; keep a simple zero tensor for compatibility
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        adata: anndata.AnnData,
        return_tensors: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        image = self._load_image_from_adata(adata)
        X_context = []
        X_cell = []

        x_pixel = adata.obs["x_pixel"].astype(int)
        y_pixel = adata.obs["y_pixel"].astype(int)

        for obs_index, x, y in zip(adata.obs_names, x_pixel, y_pixel):
            tile_ctx = self._crop_tile(image, x, y, self.context_diameter)
            t_ctx = self.transform_ctx(Image.fromarray(tile_ctx))
            X_context.append(t_ctx)
            # Provide a placeholder 56x56 zero tensor for cell view
            X_cell.append(
                torch.zeros(
                    (3, self.cell_diameter, self.cell_diameter), dtype=torch.float32
                )
            )

        patches_context = torch.stack(X_context, dim=0)
        patches_cell = torch.stack(X_cell, dim=0)

        if return_tensors == "pt":
            return {"patches_ctx": patches_context, "patches_cell": patches_cell}
        elif return_tensors is None:
            return {
                "patches_ctx": patches_context.numpy(),
                "patches_cell": patches_cell.numpy(),
            }
        else:
            raise ValueError(
                f"Unsupported return_tensors type: {return_tensors}. Use 'pt' or None."
            )

    @property
    def model_input_names(self):
        return ["patches_ctx", "patches_cell"]

    def _load_image_from_adata(
        self, adata: anndata.AnnData
    ) -> Union[Image.Image, np.ndarray]:
        if "he_slide" in adata.uns:
            return adata.uns["he_slide"]
        if "20x_slide" in adata.uns:
            return adata.uns["20x_slide"]
        if "image_path" in adata.uns:
            return Image.open(adata.uns["image_path"]).convert("RGB")
        raise ValueError(
            "Image information not available in AnnData.uns; expected 'image_path', 'he_slide', or '20x_slide'"
        )

    def _crop_tile(
        self,
        image: Union[Image.Image, np.ndarray],
        x_pixel: int,
        y_pixel: int,
        crop_diameter_pixels: int,
    ) -> np.ndarray:
        d = int(crop_diameter_pixels)
        x = x_pixel - (d // 2)
        y = y_pixel - (d // 2)

        if isinstance(image, Image.Image):
            img_w, img_h = image.size
            crop = np.array(
                image.crop((max(0, x), max(0, y), min(img_w, x + d), min(img_h, y + d)))
            )
        elif isinstance(image, np.ndarray):
            img_h, img_w = image.shape[0], image.shape[1]
            crop = image[max(0, y) : min(img_h, y + d), max(0, x) : min(img_w, x + d)]
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        actual_h, actual_w = crop.shape[:2]
        pad_y = d - actual_h
        pad_x = d - actual_w
        if pad_y > 0 or pad_x > 0:
            crop = np.pad(
                crop,
                ((0, pad_y), (0, pad_x), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        return crop[:, :, :3]
