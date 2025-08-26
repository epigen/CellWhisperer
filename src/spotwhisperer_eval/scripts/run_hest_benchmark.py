from hest.bench import benchmark
import torch
import torch.nn as nn
import os
from cellwhisperer.utils.model_io import load_cellwhisperer_model


class HESTCompatibleUNIWrapper(nn.Module):
    """
    Wrapper to make the UNI image model compatible with HEST benchmark requirements.

    HEST expects:
    - model.eval_transforms: transforms to apply to input patches
    - model.forward(patches): forward pass taking patches tensor directly
    """

    def __init__(self, uni_model, transforms):
        super().__init__()
        self.uni_model = uni_model
        self.transforms = transforms

    @property
    def eval_transforms(self):
        """Return transforms for HEST compatibility"""
        return self.transforms

    def forward(self, patches):
        """
        Forward pass compatible with HEST expectations.

        Args:
            patches: torch.Tensor of shape (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: Image embeddings
        """
        # UNI model expects (n_patches, n_scales, 3, 224, 224)
        # We need to adapt single-scale HEST patches to multi-scale UNI format

        batch_size = patches.shape[0]
        n_scales = len(self.uni_model.config.scale_factors)

        # Replicate the patches across all scales (simple approach)
        # In a more sophisticated approach, you might want to actually resize patches
        multi_scale_patches = patches.unsqueeze(1).expand(-1, n_scales, -1, -1, -1)

        # Forward through UNI model
        _, embeddings = self.uni_model(multi_scale_patches, return_dict=False)

        return embeddings


# Load CellWhisperer/SpotWhisperer model
(
    pl_model_cellwhisperer,
    text_processor_cellwhisperer,
    cellwhisperer_transcriptome_processor,
    cellwhisperer_image_processor,
) = load_cellwhisperer_model(model_path=snakemake.input.model, eval=True)

# Extract the image model and processor
cellwhisperer_model = pl_model_cellwhisperer.model
uni_image_model = cellwhisperer_model.image_model
image_transforms = cellwhisperer_image_processor.transform

# Create HEST-compatible wrapper
model = HESTCompatibleUNIWrapper(uni_image_model, image_transforms)
precision = torch.float32

print("Model loaded successfully")
print(f"Using model: {type(model)}")
print(f"Image model: {type(uni_image_model)}")

# Run benchmark
benchmark(
    model,
    model.eval_transforms,
    precision,
    config=snakemake.input.config_file,
)

# Create completion marker
with open(snakemake.output.results_marker, "w") as f:
    f.write("HEST benchmark completed successfully\n")
