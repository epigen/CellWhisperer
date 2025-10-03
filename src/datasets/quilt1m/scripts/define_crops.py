import pandas as pd
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm
import anndata as ad
import sys
from PIL import Image
import logging

# Add src to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from cellwhisperer.config import config

logging.basicConfig(level=logging.INFO)

# Load metadata
df = pd.read_csv(snakemake.input.metadata_filtered)

print(f"Processing {len(df)} images")

Path(snakemake.output.cropped_images).mkdir(parents=True, exist_ok=True)

# Process each image
for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating h5ad files"):
    image_path = Path(snakemake.input.full_res_unpacked) / row["image_path"]

    # Skip if image doesn't exist
    if not image_path.exists():
        print(f"Image {image_path} does not exist, skipping. This needs to be fixed!")
        continue

    # Get image dimensions from metadata
    h, w = int(row["height"]), int(row["width"])

    # Skip if image is smaller than crop_size
    if h < snakemake.params.crop_size or w < snakemake.params.crop_size:
        print(f"Image {image_path} is too small for crops, skipping.")
        continue

    # Get configuration for quilt1m
    he_config_name = config["dataset_he_mapping"]["quilt1m"]  # "visium_resolution"
    he_config = config["he_configs"][he_config_name]
    crop_ratio = config.get("quilt1m_config", {}).get("crop_center_ratio", 0.5)

    # Calculate center region based on config
    center_h, center_w = h // 2, w // 2
    crop_region_h, crop_region_w = int(h * crop_ratio), int(w * crop_ratio)

    # Ensure crop region is at least crop_size, but not larger than image
    crop_region_h = min(max(crop_region_h, snakemake.params.crop_size), h)
    crop_region_w = min(max(crop_region_w, snakemake.params.crop_size), w)

    start_y = center_h - crop_region_h // 2
    end_y = center_h + crop_region_h // 2
    start_x = center_w - crop_region_w // 2
    end_x = center_w + crop_region_w // 2

    # Check if image is large enough for crops
    max_y = end_y - snakemake.params.crop_size
    max_x = end_x - snakemake.params.crop_size

    if max_y < start_y or max_x < start_x:
        print(f"Image {image_path} is too small for crops, skipping.")
        continue

    # Generate random crop coordinates
    random.seed(42)  # For reproducible results
    crop_coordinates = []

    for i in range(snakemake.params.num_crops_per_image):
        y = random.randint(start_y, max_y)
        x = random.randint(start_x, max_x)
        crop_coordinates.append((x, y, i))

    # Create observation dataframe
    obs_data = pd.DataFrame(
        {
            "patch_id": [
                f"{image_path.stem}_crop_{i:03d}" for _, _, i in crop_coordinates
            ],
            "x_pixel": [x for x, _, _ in crop_coordinates],
            "y_pixel": [y for _, y, _ in crop_coordinates],
            "natural_language_annotation": [row["caption"]] * len(crop_coordinates),
        }
    )
    obs_data.index = obs_data["patch_id"]

    # Create empty var and X for h5ad structure
    var_data = pd.DataFrame(index=[])
    X = np.empty((len(crop_coordinates), 0))

    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs_data, var=var_data)

    # Store image path and metadata in uns
    adata.uns["slide_path"] = str(image_path)
    adata.uns["image_width"] = w
    adata.uns["image_height"] = h
    patch_size = he_config["patch_size_pixels"]  # 224
    adata.uns["patch_size"] = patch_size  # Use config value
    adata.uns["spot_diameter_fullres"] = snakemake.params.crop_size
    adata.uns["dataset"] = "quilt1m"
    adata.uns["modality"] = "image_text"
    adata.uns["image_path"] = image_path.as_posix()
    adata.uns["image_fn_stem"] = image_path.stem

    # Save h5ad file
    adata.write_h5ad(
        Path(snakemake.output.cropped_images) / f"full_data_{image_path.stem}.h5ad"
    )  # as per `full_dataset_multi` in config.yaml (could be passed as params)

# Generate example patches for report
logging.info(
    f"Generating {len(snakemake.output.report_patches)} example patches for QC"
)

# Collect example patches from different images
example_patches = []
patch_count = 0

for _, row in df.iterrows():
    image_path = Path(snakemake.input.full_res_unpacked) / row["image_path"]

    # Skip if we have enough patches or image doesn't exist
    if patch_count >= len(snakemake.output.report_patches) or not image_path.exists():
        continue

    # Load image and extract one random patch
    try:
        image = Image.open(image_path)
        w, h = image.size

        # Get a random crop from center region
        he_config_name = config["dataset_he_mapping"]["quilt1m"]  # "visium_resolution"
        he_config = config["he_configs"][he_config_name]
        crop_ratio = config.get("quilt1m_config", {}).get("crop_center_ratio", 0.5)

        center_h, center_w = h // 2, w // 2
        crop_region_h, crop_region_w = int(h * crop_ratio), int(w * crop_ratio)
        crop_region_h = min(max(crop_region_h, snakemake.params.crop_size), h)
        crop_region_w = min(max(crop_region_w, snakemake.params.crop_size), w)

        start_y = center_h - crop_region_h // 2
        end_y = center_h + crop_region_h // 2
        start_x = center_w - crop_region_w // 2
        end_x = center_w + crop_region_w // 2

        max_y = end_y - snakemake.params.crop_size
        max_x = end_x - snakemake.params.crop_size

        if max_y >= start_y and max_x >= start_x:
            random.seed(42 + patch_count)  # Different seed for each patch
            y = random.randint(start_y, max_y)
            x = random.randint(start_x, max_x)

            patch = image.crop(
                (x, y, x + snakemake.params.crop_size, y + snakemake.params.crop_size)
            )

            output_path = snakemake.output.report_patches[patch_count]
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            patch.save(output_path)
            logging.info(f"Saved example patch to {output_path}")
            patch_count += 1

    except Exception as e:
        logging.warning(f"Failed to extract patch from {image_path}: {e}")
        continue
