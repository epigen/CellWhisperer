import json
import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path
from tqdm import tqdm
import openslide
from PIL import Image
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

print("Creating h5ad files from PathGen data...")

# Load filtered metadata
with open(snakemake.input.metadata_filtered, "r") as f:
    data = json.load(f)
print(f"Loaded {len(data)} filtered entries")

# Create output directory
output_dir = Path(snakemake.output.h5ads)
output_dir.mkdir(parents=True, exist_ok=True)

# Group entries by WSI file for efficient processing
wsi_groups = {}
for entry in data:
    file_id = entry["file_id"]
    if file_id not in wsi_groups:
        wsi_groups[file_id] = []
    wsi_groups[file_id].append(entry)

print(f"Processing {len(wsi_groups)} WSI files")

# Process each WSI file
example_patches = []

for file_id, entries in tqdm(wsi_groups.items(), desc="Processing WSI files"):
    # Limit patches per WSI to manage output size
    if len(entries) > snakemake.params.max_patches_per_wsi:
        entries = entries[: snakemake.params.max_patches_per_wsi]
        logging.info(
            f"Limited {file_id} to {snakemake.params.max_patches_per_wsi} patches"
        )

    # Get WSI files directory from project structure
    # The download_complete file is at PROJECT_DIR / "results" / "pathgen" / ".gdc_downloads_complete"
    project_dir = Path(str(snakemake.input.download_complete)).parent.parent.parent
    wsi_files_dir = project_dir / "resources" / "pathgen" / "wsi_files"
    wsi_path = wsi_files_dir / f"{file_id}.svs"

    # Load WSI
    logging.info(f"Processing WSI: {wsi_path}")

    # Check if this is a mock file for testing (empty file)
    wsi = openslide.OpenSlide(str(wsi_path))
    wsi_width, wsi_height = wsi.dimensions

    # Create observation dataframe for this WSI
    obs_data = []
    valid_entries = 0

    for entry in entries:
        position = entry["position"]
        # PathGen coordinates are patch centers; original JSON contains top-left from TCGA extraction
        x_topleft, y_topleft = int(position[0]), int(position[1])

        # Convert to center coordinates (PathGen uses centers as primary coordinate system)
        half_patch = snakemake.params.patch_size // 2
        x_center = x_topleft + half_patch
        y_center = y_topleft + half_patch

        # Validate position is within WSI bounds (using top-left for bounds checking)
        if (
            x_topleft + snakemake.params.patch_size > wsi_width
            or y_topleft + snakemake.params.patch_size > wsi_height
            or x_topleft < 0
            or y_topleft < 0
        ):
            logging.warning(
                f"Position {x_topleft},{y_topleft} out of bounds for WSI {file_id} (size: {wsi_width}x{wsi_height})"
            )
            continue

        # Use center coordinates for patch ID and storage
        patch_id = f"{entry['wsi_id']}_x{x_center}_y{y_center}"
        obs_data.append(
            {
                "patch_id": patch_id,
                "x_pixel": x_center,  # Center coordinates (primary system)
                "y_pixel": y_center,  # Center coordinates (primary system)
                "natural_language_annotation": entry["caption"],
                "wsi_id": entry["wsi_id"],
                "file_id": file_id,
            }
        )
        valid_entries += 1

    logging.info(f"WSI {file_id}: {valid_entries}/{len(entries)} patches are valid")

    # Convert to DataFrame
    obs_df = pd.DataFrame(obs_data)
    obs_df.index = obs_df["patch_id"]

    # Create empty var and X for h5ad structure
    var_data = pd.DataFrame(index=[])
    X = np.empty((len(obs_df), 0))

    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs_df, var=var_data)

    # Store metadata in uns
    adata.uns["image_width"] = wsi_width
    adata.uns["image_height"] = wsi_height
    adata.uns["patch_size"] = snakemake.params.patch_size
    adata.uns["spot_diameter_fullres"] = snakemake.params.patch_size
    adata.uns["dataset"] = "pathgen"
    adata.uns["modality"] = "image_text"
    adata.uns["image_path"] = str(wsi_path)
    adata.uns["image_fn_stem"] = wsi_path.stem
    adata.uns["file_id"] = file_id
    adata.uns["coordinate_system"] = "center"  # Document that we use center coordinates
    adata.uns["pixel_size"] = 0.5  # We want 20x magnification
    # Save h5ad file
    output_path = output_dir / f"full_data_{file_id}.h5ad"
    adata.write_h5ad(output_path)
    logging.info(f"Saved {len(obs_df)} patches to {output_path}")

    # Generate example patches for QC report
    if len(example_patches) < len(snakemake.output.report_patches) and obs_data:
        # Extract one patch for visualization (convert center to top-left for OpenSlide)
        first_entry = obs_data[0]
        x_center, y_center = first_entry["x_pixel"], first_entry["y_pixel"]

        # Convert center coordinates to top-left for OpenSlide
        half_patch = snakemake.params.patch_size // 2
        x_topleft = x_center - half_patch
        y_topleft = y_center - half_patch

        logging.info(
            f"Extracting example patch from center coords ({x_center},{y_center})"
        )

        patch = wsi.read_region(
            (x_topleft, y_topleft),
            0,
            (snakemake.params.patch_size, snakemake.params.patch_size),
        )

        # Convert RGBA to RGB
        patch_rgb = Image.new("RGB", patch.size, (255, 255, 255))
        patch_rgb.paste(
            patch,
            mask=patch.split()[3] if len(patch.split()) == 4 else None,
        )

        output_patch_path = snakemake.output.report_patches[len(example_patches)]
        Path(output_patch_path).parent.mkdir(parents=True, exist_ok=True)
        patch_rgb.save(output_patch_path)
        example_patches.append(output_patch_path)
        logging.info(f"Saved example patch to {output_patch_path}")

    wsi.close()

# Create any remaining example patches as empty files if we don't have enough
while len(example_patches) < len(snakemake.output.report_patches):
    placeholder_path = snakemake.output.report_patches[len(example_patches)]
    Path(placeholder_path).parent.mkdir(parents=True, exist_ok=True)

    # Create a simple placeholder image
    placeholder_img = Image.new(
        "RGB",
        (snakemake.params.patch_size, snakemake.params.patch_size),
        (200, 200, 200),
    )
    placeholder_img.save(placeholder_path)
    example_patches.append(placeholder_path)
    logging.info(f"Created placeholder patch at {placeholder_path}")

print(f"Created {len(list(output_dir.glob('*.h5ad')))} h5ad files")
print(f"Generated {len(example_patches)} example patches")

logging.info("PathGen h5ad creation completed successfully")
