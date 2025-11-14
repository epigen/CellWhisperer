import torch
import torch.nn as nn
import anndata

import openslide
from tqdm import tqdm
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.processing_utils import ProcessorMixin
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import logging
import torch
from torchvision import transforms
import timm
import os

from cellwhisperer.config import config

logger = logging.getLogger(__name__)

# Constants for UNI
PAD_TOKEN_ID = 0
MODEL_INPUT_SIZE = 2048


class UNIProcessor(ProcessorMixin):
    attributes = []

    def __init__(self, config_param=None, *args, **kwargs):
        # Get H&E configuration defaults (use visium_resolution as default)
        he_config = config["he_configs"]["visium_resolution"]
        self.fallback_spot_diameter_fullres = he_config.get(
            "spot_diameter_um", 100
        )  # TODO this corresponds to visium (where spot centers are 105um apart)

        self.config = config_param or UNIConfig()  # Use default config if none provided
        # Separate transforms for context (224x224) and cell (56x56) without resizing
        self.transform_context = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        self.transform_cell = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        # Backward-compatible transform attribute used by some wrappers
        self.transform = self.transform_context

        super().__init__(*args, **kwargs)

    def __call__(
        self,
        adata: anndata.AnnData,
        return_tensors: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        image: PIL.Image
        adata: AnnData with for spot coordinates (pixel units, y,x)
        Returns: list of multiscale image patches
        """
        spot_diameter_fullres = adata.uns.get(
            "spot_diameter_fullres", self.fallback_spot_diameter_fullres
        )
        y_pixel = adata.obs.y_pixel.astype(int)
        x_pixel = adata.obs.x_pixel.astype(int)
        if adata.uns.get("dataset") == "lung_tissue":
            x_pixel, y_pixel = (
                y_pixel,
                x_pixel,
            )  # TODO temporary flip needed for LUNG datasets (i.e. best would be to fix this in the dataset)
            logging.warning(
                "Flipping x and y pixel coordinates (because I assume it's the lung dataset)"
            )
        try:
            image = adata.uns["he_slide"]
        except KeyError:
            # adata.uns["image_path"] = adata.uns["image_path"].replace(
            #     "quilt1m_lowres", "quilt1m/fullres"
            # )  # TODO drop
            try:
                image_path = adata.uns["image_path"]
                # Check file extension to determine how to load
                _, ext = os.path.splitext(image_path.lower())
                if ext in [".svs"]:
                    image = openslide.OpenSlide(image_path)
                else:
                    image = Image.open(image_path).convert("RGB")
            except FileNotFoundError as e:
                raise ValueError(
                    "Image path not available. Perhaps you need to run: `snakemake -R unpack_data` in src/datasets/quilt1m"
                )

        # Build named views: context (224x224) and cell (56x56)
        views = self.config.views
        X_context = []
        X_cell = []

        for i, (obs_index, x, y) in tqdm(
            enumerate(zip(adata.obs_names, x_pixel, y_pixel))
        ):
            # Context view
            try:
                tile_ctx = self._crop_tile(image, x, y, views["context"])
                t_ctx = self.transform_context(Image.fromarray(tile_ctx))
            except Exception as e:
                logger.error(
                    f"Error processing context tile for barcode {obs_index} at ({x}, {y}): {e}"
                )
                t_ctx = torch.zeros((3, 224, 224), dtype=torch.float32)
            X_context.append(t_ctx)

            # Cell view
            # Note: Only the context view implements physical scale anchoring via crop size.
            #       The cell view uses a fixed 56x56 pixel crop without physical anchoring.
            try:
                tile_cell = self._crop_tile(image, x, y, views["cell"])
                t_cell = self.transform_cell(Image.fromarray(tile_cell))
            except Exception as e:
                logger.error(
                    f"Error processing cell tile for barcode {obs_index} at ({x}, {y}): {e}"
                )
                t_cell = torch.zeros((3, 56, 56), dtype=torch.float32)
            X_cell.append(t_cell)

        patches_context = torch.stack(X_context, dim=0)  # (n_patches, 3, 224, 224)
        patches_cell = torch.stack(X_cell, dim=0)  # (n_patches, 3, 56, 56)

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

    def _crop_tile(
        self,
        image: Union[Image.Image, openslide.OpenSlide],
        x_pixel: int,
        y_pixel: int,
        crop_diameter_pixels: int,
    ) -> np.ndarray:
        d = int(crop_diameter_pixels)
        x = x_pixel - (d // 2)
        y = y_pixel - (d // 2)

        # Handle OpenSlide objects (SVS files)
        if isinstance(image, openslide.OpenSlide):
            img_w, img_h = image.dimensions
        elif isinstance(image, Image.Image):
            img_w, img_h = image.size[0], image.size[1]
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Clamp to image bounds
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(img_w, x + d)
        y_end = min(img_h, y + d)

        if isinstance(image, openslide.OpenSlide):
            region = image.read_region(
                (x_start, y_start), 0, (x_end - x_start, y_end - y_start)
            )
            crop = np.array(region.convert("RGB"))
        else:
            crop = np.array(image)[y_start:y_end, x_start:x_end]

        # Ensure full crop size; if smaller (out-of-bounds), signal for zero-filling
        actual_h, actual_w = crop.shape[:2]
        if actual_h < d or actual_w < d:
            raise ValueError("crop patch is larger than image bounds")

        return crop[:, :, :3]


class UNIConfig(PretrainedConfig):
    model_type = "uni2"

    def __init__(
        self,
        model_name="vit_giant_patch14_224",
        views: Dict[str, int] = {"context": 224, "cell": 56},
        cell_level_model=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.views = views
        self.embed_dim = 1536 if model_name == "vit_giant_patch14_224" else 384
        self.cell_level_model = (
            cell_level_model  # could be solved more elegantly via `views`, but whatever
        )

        # Validate views
        assert (
            isinstance(self.views, dict)
            and "context" in self.views
            and "cell" in self.views
        ), "views must contain 'context' and 'cell'"
        for k, v in self.views.items():
            assert (
                isinstance(k, str) and isinstance(v, int) and v > 0
            ), "view keys must be strings and values positive ints"


class UNIModel(PreTrainedModel):
    config_class = UNIConfig
    base_model_prefix = "uni2_model"
    is_parallelizable = False
    main_input_name = "patches_ctx"

    def __init__(self, config: UNIConfig):
        super().__init__(config)

        self.model = timm.create_model(
            config.model_name,
            num_classes=0,
            no_embed_class=True,
            dynamic_img_size=True,
            reg_tokens=8,
            depth=24 if config.model_name == "vit_giant_patch14_224" else None,
            num_heads=24 if config.model_name == "vit_giant_patch14_224" else None,
            init_values=1e-5,
            embed_dim=config.embed_dim,
            mlp_ratio=(
                2.66667 * 2 if config.model_name == "vit_giant_patch14_224" else None
            ),
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

        # Minimal CNN for 56x56 cell view and FiLM conditioning using context embedding
        self.cell_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        self.cell_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.film_gamma = nn.Linear(config.embed_dim, 128)
        self.film_beta = nn.Linear(config.embed_dim, 128)
        self.cell_proj = nn.Linear(128, config.embed_dim)

        self.post_init()

    def forward(
        self,
        patches_ctx: torch.Tensor,
        patches_cell: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # patches_ctx: (B, 3, 224, 224), patches_cell: (B, 3, 56, 56)
        ctx = patches_ctx
        cel = patches_cell
        assert ctx.ndim == 4 and ctx.shape[1:] == (
            3,
            224,
            224,
        ), f"context shape must be (B,3,224,224), got {ctx.shape}"
        assert cel.ndim == 4 and cel.shape[1:] == (
            3,
            56,
            56,
        ), f"cell shape must be (B,3,56,56), got {cel.shape}"

        # Context through ViT
        context_embeds = self.model(ctx)

        # If cell-level model is disabled, return context (Uni2) embeddings directly
        if not self.config.cell_level_model:
            if return_dict:
                raise NotImplementedError(
                    "Return dict is not implemented yet. Please use return_dict=False for now."
                )
            else:
                return (None, context_embeds)

        # Cell through CNN with FiLM conditioning
        feat = self.cell_cnn(cel)  # (B,128,H,W)
        gamma = (
            self.film_gamma(context_embeds).unsqueeze(-1).unsqueeze(-1)
        )  # (B,128,1,1)
        beta = self.film_beta(context_embeds).unsqueeze(-1).unsqueeze(-1)  # (B,128,1,1)
        feat = feat * gamma + beta
        pooled = self.cell_pool(feat).squeeze(-1).squeeze(-1)  # (B,128)
        cell_embeds = self.cell_proj(pooled)  # (B, embed_dim)

        if return_dict:
            raise NotImplementedError(
                "Return dict is not implemented yet. Please use return_dict=False for now."
            )
        else:
            return (None, cell_embeds)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ) -> PreTrainedModel:
        if "config" in kwargs:
            config = kwargs.pop("config")
            if isinstance(config, dict):
                config = UNIConfig(**config)
            elif not isinstance(config, UNIConfig):
                raise ValueError(
                    "Parameter `config` must be a dictionary or an instance of `UNIConfig`."
                )
        else:
            config = UNIConfig()
            logger.warning(
                "No configuration provided. Using default configuration from checkpoint."
            )
        model = cls(config, *args, **kwargs)

        # Only load pretrained weights if not uni_small
        if config.model_name != "vit_small_patch16_224":
            model.model.load_state_dict(
                torch.load(pretrained_model_name_or_path, map_location="cpu"),
                strict=True,
            )

        return model
