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
from .cell_model import CellLevelModel

logger = logging.getLogger(__name__)

# Constants for UNI
PAD_TOKEN_ID = 0
MODEL_INPUT_SIZE = 2048


class UNIProcessor(ProcessorMixin):
    attributes = []

    def __init__(self, *args, **kwargs):
        # Get H&E configuration defaults (use visium_resolution as default)
        he_config = config["he_configs"]["visium_resolution"]
        self.fallback_spot_diameter_fullres = he_config.get(
            "spot_diameter_um", 100
        )  # TODO this corresponds to visium (where spot centers are 105um apart)

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
        if "he_slide" in adata.uns:
            image = adata.uns["he_slide"]
        elif "20x_slide" in adata.uns:
            image = adata.uns["20x_slide"]  # legacy for HEST
        elif "image_path" in adata.uns:
            # adata.uns["image_path"] = adata.uns["image_path"].replace(
            #     "quilt1m_lowres", "quilt1m/fullres"
            # )  # TODO drop
            image_path = adata.uns["image_path"]
            # Check file extension to determine how to load
            _, ext = os.path.splitext(image_path.lower())
            if ext in [".svs"]:
                image = openslide.OpenSlide(image_path)
            else:
                image = Image.open(image_path).convert("RGB")

        else:
            raise ValueError(
                "Image information not available. Perhaps you need to run something like: `snakemake -R unpack_data` in src/datasets/quilt1m?"
            )

        # Build named views: context (224x224) and cell (56x56)

        X_context = []
        X_cell = []

        for i, (obs_index, x, y) in tqdm(
            enumerate(zip(adata.obs_names, x_pixel, y_pixel)),
            desc="Extracting UNI patches",
            total=len(adata.obs),
        ):
            # Context view (only if context model is enabled)
            try:
                tile_ctx = self._crop_tile(image, x, y, UNIConfig().views["context"])
                t_ctx = self.transform_context(Image.fromarray(tile_ctx))
            except (IndexError, ValueError) as e:
                logger.error(
                    f"Error processing context tile for barcode {obs_index} at ({x}, {y}): {e}"
                )
                t_ctx = torch.zeros((3, 224, 224), dtype=torch.float32)
            X_context.append(t_ctx)

            # Cell view (only if cell level model is enabled)
            # Note: Only the context view implements physical scale anchoring via crop size.
            #       The cell view uses a fixed 56x56 pixel crop without physical anchoring.
            try:
                tile_cell = self._crop_tile(image, x, y, UNIConfig().views["cell"])
                t_cell = self.transform_cell(Image.fromarray(tile_cell))
            except (IndexError, ValueError) as e:
                logger.error(
                    f"Error processing cell tile for barcode {obs_index} at ({x}, {y}): {e}"
                )
                t_cell = torch.zeros((3, 56, 56), dtype=torch.float32)
            X_cell.append(t_cell)

        # Create tensors only for enabled models
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
        elif isinstance(image, np.ndarray):
            img_h, img_w = image.shape[0], image.shape[1]
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
        elif isinstance(image, np.ndarray):
            crop = image[y_start:y_end, x_start:x_end]
        elif isinstance(image, Image.Image):
            crop = np.array(image.crop((x_start, y_start, x_end, y_end)))
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
        cell_level_model=False,
        context_model=True,
        cnn_embedding_dim=128,
        cnn_num_layers=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.views = views
        self.embed_dim = 1536 if model_name == "vit_giant_patch14_224" else 384
        self.cell_level_model = bool(
            cell_level_model  # could be solved more elegantly via `views`, but whatever
        )
        self.context_model = context_model
        self.cnn_embedding_dim = cnn_embedding_dim
        self.cnn_num_layers = cnn_num_layers

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

        # Validate configuration: at least one model must be enabled
        assert (
            self.cell_level_model or self.context_model
        ), "At least one of cell_level_model or context_model must be True"

        # assert that views are not modified (only default supported at the moment because we use it in the Processor)
        assert self.views == {
            "context": 224,
            "cell": 56,
        }, "Only default views {'context':224, 'cell':56} are supported currently"


class UNIModel(PreTrainedModel):
    config_class = UNIConfig
    base_model_prefix = "uni2_model"
    is_parallelizable = False
    main_input_name = "patches_ctx"

    def __init__(self, config: UNIConfig):
        super().__init__(config)

        # Only create ViT model if context model is enabled
        if getattr(config, "context_model", True):
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
                    2.66667 * 2
                    if config.model_name == "vit_giant_patch14_224"
                    else None
                ),
                mlp_layer=timm.layers.SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        else:
            self.model = None

        # Cell-level model for 56x56 cell view
        if getattr(config, "cell_level_model", False):
            if config.context_model:
                # FiLM-conditioned cell model
                self.cell_model = CellLevelModel.create_film_conditioned(
                    context_dim=config.embed_dim,
                    embedding_dim=config.cnn_embedding_dim,
                    num_layers=config.cnn_num_layers,
                    output_dim=config.embed_dim,
                )
            else:
                # Standalone cell model
                self.cell_model = CellLevelModel.create_standalone(
                    embedding_dim=config.cnn_embedding_dim,
                    num_layers=config.cnn_num_layers,
                    output_dim=config.embed_dim,
                )
        else:
            self.cell_model = None

        self.post_init()

    def forward(
        self,
        patches_ctx: torch.Tensor,
        patches_cell: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # patches_ctx: (B, 3, 224, 224), patches_cell: (B, 3, 56, 56)

        # Return the appropriate embeddings based on configuration
        context_enabled = getattr(self.config, "context_model", True)
        cell_enabled = getattr(self.config, "cell_level_model", False)

        # Context through ViT (if enabled)
        if context_enabled:
            assert (
                patches_ctx is not None
                and patches_ctx.ndim == 4
                and patches_ctx.shape[1:]
                == (
                    3,
                    224,
                    224,
                )
            ), f"context shape must be (B,3,224,224), got {patches_ctx.shape}"

            context_embeds = self.model(patches_ctx)

        # Cell through CNN (if enabled)
        if cell_enabled:
            raise ValueError("this should be disabled for now")

            assert (
                patches_cell is not None
                and patches_cell.ndim == 4
                and patches_cell.shape[1:]
                == (
                    3,
                    56,
                    56,
                )
            ), f"cell shape must be (B,3,56,56), got {patches_cell.shape}"

            cell_embeds = self.cell_model(patches_cell, context_embeds)

        if context_enabled and cell_enabled:
            # Both models enabled: return cell embeddings (FiLM-conditioned)
            output_embeds = cell_embeds
        elif context_enabled and not cell_enabled:
            # Only context model: return context embeddings
            output_embeds = context_embeds
        elif not context_enabled and cell_enabled:
            # Only cell model: return cell embeddings (no FiLM conditioning)
            output_embeds = cell_embeds
        else:
            # This should never happen due to validation in config
            raise ValueError(
                "At least one of context_model or cell_level_model must be enabled"
            )

        if return_dict:
            raise NotImplementedError(
                "Return dict is not implemented yet. Please use return_dict=False for now."
            )
        else:
            return (None, output_embeds)

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

        # Only load pretrained weights if not uni_small and context model is enabled
        if config.model_name != "vit_small_patch16_224" and config.context_model:
            model.model.load_state_dict(
                torch.load(pretrained_model_name_or_path, map_location="cpu"),
                strict=True,
            )

        return model
