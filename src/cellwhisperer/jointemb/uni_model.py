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
        self.fallback_spot_diameter_fullres = he_config.get("spot_diameter_um", 100)
        patch_size = he_config["patch_size_pixels"]

        self.config = config_param or UNIConfig()  # Use default config if none provided
        self.transform = transforms.Compose(
            [
                transforms.Resize(patch_size),
                transforms.CenterCrop(patch_size),
                transforms.ToTensor(),
                transforms.Normalize(  # TODO understand better. not sure if this is too good tbh...
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

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
            image = adata.uns["20x_slide"]
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

        X = []
        # Use scale factors from config
        scale_factors = self.config.scale_factors

        for i, (obs_index, x, y) in tqdm(
            enumerate(zip(adata.obs_names, x_pixel, y_pixel))
        ):
            multi_scale_tiles = []

            for scale_factor in scale_factors:
                try:
                    tile = self._crop_tile(
                        image, x, y, spot_diameter_fullres, scale_factor
                    )
                    tile = self.transform(
                        Image.fromarray(tile)
                    )  # transform is defined below
                except Exception as e:
                    logger.error(
                        f"Error processing tile for barcode {obs_index} at ({x}, {y}) with scale {scale_factor}: {e}"
                    )
                    tile = torch.zeros(
                        (3, 224, 224), dtype=torch.float32
                    )  # Fallback to zero tensor if transformation fails
                multi_scale_tiles.append(tile)

                # To check how the crops look like
                # if scale_factor == 1.0:
                #     import os

                #     crop_dir = "/scratch/users/moritzs/scale1crops"
                #     os.makedirs(crop_dir, exist_ok=True)
                #     crop_path = os.path.join(crop_dir, f"{obs_index}_scale1.png")
                #     Image.fromarray(
                #         self._crop_tile(
                #             image, x, y, spot_diameter_fullres, scale_factor
                #         )
                #     ).save(crop_path)

            # Stack the scales for this patch
            multi_scale_patch = torch.stack(
                multi_scale_tiles, dim=0
            )  # (n_scales, 3, 224, 224)
            X.append(multi_scale_patch)

        X = torch.stack(
            X, dim=0
        )  # (n_patches, n_scales, 3, 224, 224) - first dim: patches, second: scale levels, third: RGB channels

        if return_tensors == "pt":
            return {
                "patches": X  # Already a torch tensor (n_patches, n_scales, 3, 224, 224)
            }
        elif return_tensors is None:
            return {"patches": X.numpy()}
        else:
            raise ValueError(
                f"Unsupported return_tensors type: {return_tensors}. Use 'pt' or None."
            )

    @property
    def model_input_names(self):
        return ["patches"]

    def _crop_tile(
        self,
        image: Union[Image.Image, openslide.OpenSlide],
        x_pixel,
        y_pixel,
        cell_diameter_pixels,
        scale_factor=1.0,
    ) -> np.ndarray:
        scaled_diameter = int(cell_diameter_pixels * scale_factor)
        x = x_pixel - int(scaled_diameter // 2)
        y = y_pixel - int(scaled_diameter // 2)

        # Handle OpenSlide objects (SVS files)
        if isinstance(image, openslide.OpenSlide):
            img_w, img_h = image.dimensions
        elif isinstance(image, Image.Image):
            img_h, img_w = image.size[1], image.size[0]
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        if scale_factor != 1.0:
            # For non-1.0 scale factors, move the crop to stay within image bounds
            # while retaining the size

            # Adjust x coordinates to stay within bounds
            if x < 0:
                x = 0
            elif x + scaled_diameter > img_w:
                x = img_w - scaled_diameter

            # Adjust y coordinates to stay within bounds
            if y < 0:
                y = 0
            elif y + scaled_diameter > img_h:
                y = img_h - scaled_diameter

            # Final bounds check - if the scaled diameter is larger than the image,
            # we still need to crop what we can
            x_start = max(0, x)
            y_start = max(0, y)
            x_end = min(img_w, x + scaled_diameter)
            y_end = min(img_h, y + scaled_diameter)

            if isinstance(image, openslide.OpenSlide):
                # Read region directly from OpenSlide
                region_width = x_end - x_start
                region_height = y_end - y_start
                region = image.read_region(
                    (x_start, y_start), 0, (region_width, region_height)
                )
                cell = np.array(region.convert("RGB"))
            elif isinstance(image, Image.Image):
                cell = np.array(image)[y_start:y_end, x_start:x_end]

            # If the crop is still smaller than expected, return None to indicate zero tensor needed
            actual_h, actual_w = cell.shape[:2]
            if actual_h < scaled_diameter or actual_w < scaled_diameter:
                raise ValueError("scaled patch is larger than image bounds")
        else:
            # For scale_factor == 1.0, return zero tensor if patch is not fully within image
            # Calculate the bounds
            y_start, y_end = y, y + scaled_diameter
            x_start, x_end = x, x + scaled_diameter

            # Check if patch is fully within the image
            if y_start < 0 or y_end > img_h or x_start < 0 or x_end > img_w:
                # Return None to indicate zero tensor needed
                raise ValueError("Patch (scale factor=1.0) is out of image bounds")
            else:
                # Crop the image (patch is fully within bounds)
                if isinstance(image, openslide.OpenSlide):
                    region = image.read_region(
                        (x_start, y_start), 0, (scaled_diameter, scaled_diameter)
                    )
                    cell = np.array(region.convert("RGB"))
                elif isinstance(image, Image.Image):
                    cell = np.array(image)[y_start:y_end, x_start:x_end]

        return cell[:, :, :3]


class UNIConfig(PretrainedConfig):
    model_type = "uni2"

    def __init__(
        self,
        model_name="vit_giant_patch14_224",
        scale_factors=[
            1.0,
        ],  # quilt1m and hest1k data was computed with 0.6, 1.0, 4.0
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.scale_factors = scale_factors
        self.embed_dim = None

        # Ensure one of the scale factors is 1.0
        assert 1.0 in scale_factors, "One of the scale factors must be 1.0"
        assert (
            len(scale_factors) == 1
        ), "Currently only single scale factor is supported"


class UNIModel(PreTrainedModel):
    config_class = UNIConfig
    base_model_prefix = "uni2_model"
    is_parallelizable = False
    main_input_name = "patches"

    def __init__(self, config: UNIConfig):
        super().__init__(config)

        self.model = timm.create_model(
            config.model_name,
            num_classes=0,
            no_embed_class=True,
            dynamic_img_size=True,
            depth=24 if config.model_name == "vit_giant_patch14_224" else None,
            num_heads=24 if config.model_name == "vit_giant_patch14_224" else None,
            init_values=1e-5,
            embed_dim=1536 if config.model_name == "vit_giant_patch14_224" else None,
            mlp_ratio=(
                2.66667 * 2 if config.model_name == "vit_giant_patch14_224" else None
            ),
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

        # Store the embedding dimension from the model
        self.embed_dim = self.model.num_features

        self.post_init()

    def forward(
        self,
        patches: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # patches: (n_patches, n_scales, 3, 224, 224) - first dim: patches, second: scale levels, third: RGB channels
        n_scales = len(self.config.scale_factors)
        assert (
            patches.ndim == 5  # (B, n_scales, 3, 224, 224)
            and patches.shape[1] == n_scales  # number of scales
            and patches.shape[2] == 3  # RGB channels
            and patches.shape[3:] == (224, 224)
        ), f"Expected input shape (n_patches, {n_scales}, 3, 224, 224), but got {patches.shape}"

        n_patches, n_scales, channels, height, width = patches.shape

        # Process each scale separately
        scale_embeddings = []
        for scale_idx in range(n_scales):
            # Extract patches for this scale: (n_patches, 3, 224, 224)
            scale_patches = patches[:, scale_idx, :, :, :]

            # Process through UNI model
            scale_outputs = self.model(
                scale_patches
            )  # TODO check for the giant model! Does hidden_features work there?
            scale_embeddings.append(scale_outputs)

        # Concatenate embeddings from all scales
        # Each scale_outputs has shape (n_patches, embed_dim), so concat along embed_dim
        concatenated_outputs = torch.cat(
            scale_embeddings, dim=-1
        )  # (n_patches, n_scales * embed_dim)

        if return_dict:
            raise NotImplementedError(
                "Return dict is not implemented yet. Please use return_dict=False for now."
            )
        else:
            return (None, concatenated_outputs)

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
