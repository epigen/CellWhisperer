import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.processing_utils import ProcessorMixin
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import logging
import timm
import torch
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download


logger = logging.getLogger(__name__)

# Constants for UNI
PAD_TOKEN_ID = 0
MODEL_INPUT_SIZE = 2048


class UNIProcessor(ProcessorMixin):
    attributes = []

    def __init__(self, *args, **kwargs):
        self.fallback_spot_diameter_fullres = 100
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(  # TODO understand
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        super().__init__(*args, **kwargs)

    def __call__(
        self,
        adata,
        return_tensors: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        image: PIL.Image
        adata: AnnData with .obsm[spatial_key] for spot coordinates (pixel units, y,x)
        transcriptome_key: adata.X or adata.layers[key]
        spatial_key: key in adata.obsm for spot positions
        Returns: list of (image_patch, transcriptome) pairs
        """
        spot_diameter_fullres = adata.uns.get(
            "spot_diameter_fullres", self.fallback_spot_diameter_fullres
        )
        barcodes = adata.obs_names
        x_pixel = adata.obs.x_pixel
        y_pixel = adata.obs.y_pixel
        image = adata.uns["20x_slide"]
        if isinstance(image, Image.Image):
            raise NotImplementedError(
                "Currently, only numpy arrays are supported for image input."
            )
        X = []

        for i, (b, x, y) in tqdm(enumerate(zip(barcodes, x_pixel, y_pixel))):
            main_tile = self._crop_tile(image, x, y, spot_diameter_fullres)  #

            try:
                main_tile = self.transform(
                    Image.fromarray(main_tile)
                )  # transform is defined below below
            except Exception as e:
                logger.error(
                    f"Error processing tile for barcode {b} at ({x}, {y}): {e}"
                )
                main_tile = np.zeros(
                    (3, 224, 224), dtype=np.float32
                )  # Fallback to zero tensor if transformation fails
            X.append(main_tile)

        X = np.stack(X, axis=0)  # (n_patches, 3, 224, 224)

        if return_tensors == "pt":
            return {
                "patches": torch.tensor(
                    X, dtype=torch.float32
                )  # (n_patches, 3, 224, 224)
            }
        elif return_tensors is None:
            return {"patches": X}
        else:
            raise ValueError(
                f"Unsupported return_tensors type: {return_tensors}. Use 'pt' or None."
            )

    @property
    def model_input_names(self):
        return ["patches"]

    def _crop_tile(self, image: np.ndarray, x_pixel, y_pixel, cell_diameter_pixels):
        # NOTE: implementation for Image.Image is in test_uni.ipynb
        x = x_pixel - int(cell_diameter_pixels // 2)
        y = y_pixel - int(cell_diameter_pixels // 2)
        cell = image[
            y : y + int(cell_diameter_pixels), x : x + int(cell_diameter_pixels)
        ]
        return cell[:, :, :3]


class UNIConfig(PretrainedConfig):
    model_type = "uni2"

    def __init__(
        self,
        model_name="vit_giant_patch14_224",
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        reg_tokens=8,
        dynamic_img_size=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.depth = depth
        self.num_heads = num_heads
        self.init_values = init_values
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        self.no_embed_class = no_embed_class
        self.reg_tokens = reg_tokens
        self.dynamic_img_size = dynamic_img_size


class UNIModel(PreTrainedModel):
    config_class = UNIConfig
    base_model_prefix = "uni2_model"
    is_parallelizable = False
    main_input_name = "patches"

    def __init__(self, config: UNIConfig):
        super().__init__(config)

        self.model = timm.create_model(
            config.model_name,
            img_size=config.img_size,
            patch_size=config.patch_size,
            depth=config.depth,
            num_heads=config.num_heads,
            init_values=config.init_values,
            embed_dim=config.embed_dim,
            mlp_ratio=config.mlp_ratio,
            num_classes=config.num_classes,
            no_embed_class=config.no_embed_class,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=config.reg_tokens,
            dynamic_img_size=config.dynamic_img_size,
        )

        self.post_init()

    def forward(
        self,
        patches: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # patches: (n_patches, 3, 224, 224)
        assert (
            patches.ndim == 4
            and patches.shape[1] == 3
            and patches.shape[2:] == (224, 224)
        ), f"Expected input shape (n_patches, 3, 224, 224), but got {patches.shape}"

        outputs = self.model(patches)

        if return_dict:
            raise NotImplementedError(
                "Return dict is not implemented yet. Please use return_dict=False for now."
            )
        else:
            return (None, outputs)

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

        model.model.load_state_dict(
            torch.load(pretrained_model_name_or_path, map_location="cpu"), strict=True
        )

        return model
