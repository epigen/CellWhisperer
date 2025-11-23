"""
Extract metadata from SVS files and save in human-readable format.

This script extracts technical specifications from TCGA whole slide images including:
- Magnification/objective power
- Pixel size (micrometers per pixel)
- Image dimensions at different resolution levels
- Scanner vendor and format information
- Compression details

The metadata is saved in both JSON and human-readable text formats.
"""

import json
import pandas as pd
import openslide
from pathlib import Path
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def extract_aperio_properties(slide):
    """
    Extract Aperio-specific properties from slide metadata.

    Args:
        slide: OpenSlide object

    Returns:
        dict: Aperio-specific metadata
    """
    aperio_props = {}

    # Extract all properties that start with 'aperio.'
    for key, value in slide.properties.items():
        if key.startswith("aperio."):
            aperio_props[key] = value

    return aperio_props


def extract_svs_metadata(svs_path):
    """
    Extract comprehensive metadata from an SVS file.

    Args:
        svs_path: Path to SVS file

    Returns:
        dict: Complete metadata dictionary
    """
    logging.info(f"Extracting metadata from: {svs_path}")

    slide = openslide.OpenSlide(str(svs_path))

    # Basic slide properties
    metadata = {
        "file_path": str(svs_path),
        "file_size_bytes": svs_path.stat().st_size,
        "extraction_time": datetime.now().isoformat(),
        "format_vendor": slide.detect_format(svs_path) or "unknown",
    }

    # Slide dimensions and levels
    metadata.update(
        {
            "level_count": slide.level_count,
            "dimensions_level0": slide.dimensions,
            "level_dimensions": slide.level_dimensions,
            "level_downsamples": slide.level_downsamples,
        }
    )

    # Standard OpenSlide properties
    standard_props = {}
    prop_mapping = {
        openslide.PROPERTY_NAME_VENDOR: "vendor",
        openslide.PROPERTY_NAME_QUICKHASH1: "quickhash1",
        openslide.PROPERTY_NAME_BACKGROUND_COLOR: "background_color",
        openslide.PROPERTY_NAME_OBJECTIVE_POWER: "objective_power",
        openslide.PROPERTY_NAME_MPP_X: "microns_per_pixel_x",
        openslide.PROPERTY_NAME_MPP_Y: "microns_per_pixel_y",
        openslide.PROPERTY_NAME_BOUNDS_X: "bounds_x",
        openslide.PROPERTY_NAME_BOUNDS_Y: "bounds_y",
        openslide.PROPERTY_NAME_BOUNDS_WIDTH: "bounds_width",
        openslide.PROPERTY_NAME_BOUNDS_HEIGHT: "bounds_height",
        openslide.PROPERTY_NAME_COMMENT: "comment",
    }

    for prop_name, friendly_name in prop_mapping.items():
        if prop_name in slide.properties:
            value = slide.properties[prop_name]
            # Try to convert numeric values
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string
            standard_props[friendly_name] = value

    metadata["standard_properties"] = standard_props

    # Extract Aperio-specific properties
    metadata["aperio_properties"] = extract_aperio_properties(slide)

    # Associated images
    metadata["associated_images"] = list(slide.associated_images.keys())

    # All raw properties for completeness
    metadata["all_properties"] = dict(slide.properties)

    # Calculate physical dimensions
    if (
        "microns_per_pixel_x" in standard_props
        and "microns_per_pixel_y" in standard_props
    ):
        width_pixels, height_pixels = slide.dimensions
        mpp_x = standard_props["microns_per_pixel_x"]
        mpp_y = standard_props["microns_per_pixel_y"]

        metadata["physical_dimensions"] = {
            "width_um": width_pixels * mpp_x,
            "height_um": height_pixels * mpp_y,
            "area_um2": width_pixels * mpp_x * height_pixels * mpp_y,
            "area_mm2": (width_pixels * mpp_x * height_pixels * mpp_y) / 1_000_000,
        }

    slide.close()
    logging.info(f"Successfully extracted metadata for {svs_path.name}")
    return metadata


def format_metadata_for_humans(metadata):
    """
    Format metadata dictionary into human-readable text.

    Args:
        metadata: Metadata dictionary

    Returns:
        str: Human-readable formatted text
    """
    if "error" in metadata:
        return f"ERROR: {metadata['error']}\n"

    lines = []
    lines.append("=" * 80)
    lines.append(f"SVS FILE METADATA: {Path(metadata['file_path']).name}")
    lines.append("=" * 80)
    lines.append(f"Extraction Time: {metadata['extraction_time']}")
    lines.append(
        f"File Size: {metadata['file_size_bytes']:,} bytes ({metadata['file_size_bytes'] / 1024 / 1024:.1f} MB)"
    )
    lines.append(f"Format Vendor: {metadata.get('format_vendor', 'unknown')}")
    lines.append("")

    # Magnification and pixel size
    lines.append("SCANNING SPECIFICATIONS:")
    lines.append("-" * 40)
    std_props = metadata.get("standard_properties", {})

    objective_power = std_props.get("objective_power", "unknown")
    lines.append(f"Objective Power: {objective_power}x")

    mpp_x = std_props.get("microns_per_pixel_x", "unknown")
    mpp_y = std_props.get("microns_per_pixel_y", "unknown")
    lines.append(f"Microns per Pixel X: {mpp_x}")
    lines.append(f"Microns per Pixel Y: {mpp_y}")

    if mpp_x != "unknown" and mpp_y != "unknown":
        avg_mpp = (mpp_x + mpp_y) / 2
        lines.append(f"Average Pixel Size: {avg_mpp:.3f} μm/pixel")
    lines.append("")

    # Image dimensions
    lines.append("IMAGE DIMENSIONS:")
    lines.append("-" * 40)
    lines.append(f"Level Count: {metadata.get('level_count', 'unknown')}")

    level0_dims = metadata.get("dimensions_level0", (0, 0))
    lines.append(
        f"Level 0 (Full Resolution): {level0_dims[0]:,} × {level0_dims[1]:,} pixels"
    )

    if "physical_dimensions" in metadata:
        phys = metadata["physical_dimensions"]
        lines.append(
            f"Physical Size: {phys['width_um']:.1f} × {phys['height_um']:.1f} μm"
        )
        lines.append(f"Physical Area: {phys['area_mm2']:.2f} mm²")
    lines.append("")

    # Multi-resolution levels
    if "level_dimensions" in metadata and len(metadata["level_dimensions"]) > 1:
        lines.append("PYRAMID LEVELS:")
        lines.append("-" * 40)
        for i, (dims, downsample) in enumerate(
            zip(metadata["level_dimensions"], metadata["level_downsamples"])
        ):
            lines.append(
                f"Level {i}: {dims[0]:,} × {dims[1]:,} pixels (downsample: {downsample:.1f}x)"
            )
        lines.append("")

    # Associated images
    if metadata.get("associated_images"):
        lines.append("ASSOCIATED IMAGES:")
        lines.append("-" * 40)
        for img_name in metadata["associated_images"]:
            lines.append(f"- {img_name}")
        lines.append("")

    # Aperio-specific info
    aperio_props = metadata.get("aperio_properties", {})
    if aperio_props:
        lines.append("APERIO-SPECIFIC PROPERTIES:")
        lines.append("-" * 40)
        for key, value in sorted(aperio_props.items()):
            clean_key = key.replace("aperio.", "")
            lines.append(f"{clean_key}: {value}")
        lines.append("")

    # Vendor info
    lines.append("VENDOR INFORMATION:")
    lines.append("-" * 40)
    vendor = std_props.get("vendor", "unknown")
    lines.append(f"Scanner Vendor: {vendor}")

    if "quickhash1" in std_props:
        lines.append(f"Quickhash1: {std_props['quickhash1']}")

    if "background_color" in std_props:
        lines.append(f"Background Color: {std_props['background_color']}")
    lines.append("")

    return "\n".join(lines)


def main():
    """Main processing function."""
    print("Extracting SVS metadata for diagnostic purposes...")

    # Get WSI files directory - should be PROJECT_DIR / "resources" / "pathgen" / "wsi_files"
    # The download_complete file is at PROJECT_DIR / "results" / "pathgen" / ".gdc_downloads_complete"
    project_dir = Path(snakemake.input.download_complete).parent.parent.parent
    wsi_files_dir = project_dir / "resources" / "pathgen" / "wsi_files"

    if not wsi_files_dir.exists():
        logging.error(f"WSI files directory not found: {wsi_files_dir}")
        # Create empty output files
        Path(snakemake.output.metadata_json).parent.mkdir(parents=True, exist_ok=True)
        with open(snakemake.output.metadata_json, "w") as f:
            json.dump([], f)
        with open(snakemake.output.metadata_txt, "w") as f:
            f.write("No WSI files found for metadata extraction.\n")
        return

    # Find all SVS files
    svs_files = list(wsi_files_dir.glob("*.svs"))

    logging.info(f"Found {len(svs_files)} SVS files to process")

    # Extract metadata from all files
    all_metadata = []
    human_readable_parts = []

    for svs_file in sorted(svs_files):
        metadata = extract_svs_metadata(svs_file)
        all_metadata.append(metadata)

        human_text = format_metadata_for_humans(metadata)
        human_readable_parts.append(human_text)

    # Ensure output directories exist
    Path(snakemake.output.metadata_json).parent.mkdir(parents=True, exist_ok=True)
    Path(snakemake.output.metadata_txt).parent.mkdir(parents=True, exist_ok=True)

    # Save JSON metadata
    with open(snakemake.output.metadata_json, "w") as f:
        json.dump(all_metadata, f, indent=2)

    # Save human-readable metadata
    with open(snakemake.output.metadata_txt, "w") as f:
        f.write(f"SVS METADATA DIAGNOSTIC REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total Files Processed: {len(svs_files)}\n")
        f.write("\n")

        # Summary statistics
        successful_extractions = [m for m in all_metadata if "error" not in m]
        if successful_extractions:
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 50 + "\n")

            # Magnifications
            magnifications = []
            pixel_sizes = []
            for m in successful_extractions:
                std_props = m.get("standard_properties", {})
                if "objective_power" in std_props:
                    magnifications.append(std_props["objective_power"])
                if (
                    "microns_per_pixel_x" in std_props
                    and "microns_per_pixel_y" in std_props
                ):
                    avg_mpp = (
                        std_props["microns_per_pixel_x"]
                        + std_props["microns_per_pixel_y"]
                    ) / 2
                    pixel_sizes.append(avg_mpp)

            if magnifications:
                unique_mags = set(magnifications)
                f.write(f"Magnifications found: {sorted(unique_mags)}\n")
                for mag in sorted(unique_mags):
                    count = magnifications.count(mag)
                    f.write(f"  {mag}x: {count} files\n")

            if pixel_sizes:
                avg_pixel_size = sum(pixel_sizes) / len(pixel_sizes)
                min_pixel_size = min(pixel_sizes)
                max_pixel_size = max(pixel_sizes)
                f.write(
                    f"Pixel Sizes: avg={avg_pixel_size:.3f}, min={min_pixel_size:.3f}, max={max_pixel_size:.3f} μm/pixel\n"
                )

            f.write("\n")

        # Detailed per-file information
        f.write("\n".join(human_readable_parts))

    # Create summary CSV for easy analysis
    if successful_extractions:
        csv_path = Path(snakemake.output.metadata_txt).with_suffix(".csv")

        summary_data = []
        for metadata in successful_extractions:
            std_props = metadata.get("standard_properties", {})

            row = {
                "filename": Path(metadata["file_path"]).name,
                "file_size_mb": metadata["file_size_bytes"] / 1024 / 1024,
                "objective_power": std_props.get("objective_power", None),
                "microns_per_pixel_x": std_props.get("microns_per_pixel_x", None),
                "microns_per_pixel_y": std_props.get("microns_per_pixel_y", None),
                "width_pixels": metadata.get("dimensions_level0", [None])[0],
                "height_pixels": metadata.get("dimensions_level0", [None, None])[1],
                "level_count": metadata.get("level_count", None),
                "vendor": std_props.get("vendor", None),
                "format_vendor": metadata.get("format_vendor", None),
            }

            # Add physical dimensions if available
            if "physical_dimensions" in metadata:
                phys = metadata["physical_dimensions"]
                row.update(
                    {
                        "width_um": phys["width_um"],
                        "height_um": phys["height_um"],
                        "area_mm2": phys["area_mm2"],
                    }
                )

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_path, index=False)
        logging.info(f"Saved summary CSV: {csv_path}")

    logging.info(f"Metadata extraction complete. Processed {len(svs_files)} files.")
    logging.info(f"Results saved to:")
    logging.info(f"  JSON: {snakemake.output.metadata_json}")
    logging.info(f"  Human-readable: {snakemake.output.metadata_txt}")


if __name__ == "__main__":
    main()
