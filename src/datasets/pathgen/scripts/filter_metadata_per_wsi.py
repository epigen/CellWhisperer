"""
Filter PathGen metadata and create individual JSON files per WSI.

This script loads the large PathGen metadata JSON once, filters by available WSI files,
and creates individual JSON files for each WSI containing only the patches for that WSI.
This dramatically improves efficiency for downstream processing.
"""

import json
import pandas as pd
from pathlib import Path

print("Filtering PathGen metadata and creating per-WSI JSON files...")

# Load metadata JSON
with open(snakemake.input.metadata_json, "r") as f:
    data = json.load(f)
print(f"Loaded {len(data)} entries from metadata JSON")

# Load filtered file IDs (already excludes broken datasets)
with open(snakemake.input.file_ids_list, "r") as f:
    allowed_file_ids = set(line.strip() for line in f if line.strip())
print(f"Loaded {len(allowed_file_ids)} allowed file IDs (excluding broken datasets)")

# Verify WSI files exist for allowed file IDs
wsi_dir = Path(snakemake.params.wsi_dir)
available_files = set(f.stem for f in wsi_dir.glob("*.svs"))
print(f"Found {len(available_files)} available WSI files in directory")

# Final file IDs are the intersection of allowed and available
final_file_ids = allowed_file_ids & available_files
print(f"Final file IDs to process: {len(final_file_ids)} (intersection of allowed & available)")

if len(final_file_ids) < len(allowed_file_ids):
    missing_files = allowed_file_ids - available_files
    print(f"WARNING: {len(missing_files)} allowed files not found in WSI directory:")
    for file_id in sorted(missing_files):
        print(f"  Missing: {file_id}")

# Filter metadata based on final file IDs
filtered_data = []
for entry in data:
    file_id = entry.get("file_id", "")
    if file_id and file_id in final_file_ids:
        filtered_data.append(entry)

print(f"Filtered to {len(filtered_data)} entries with available and non-broken WSI files")

# Group entries by file_id
wsi_groups = {}
for entry in filtered_data:
    file_id = entry["file_id"]
    if file_id not in wsi_groups:
        wsi_groups[file_id] = []
    wsi_groups[file_id].append(entry)

print(f"Grouped into {len(wsi_groups)} WSI groups")

# Apply testing mode limitations if needed
if snakemake.params.testing_mode:
    print("TESTING MODE: Applying limitations")
    
    # Take only first 2 WSIs and limit patches per WSI
    limited_wsi_groups = {}
    for i, (file_id, entries) in enumerate(wsi_groups.items()):
        if i >= 2:  # Only first 2 WSIs in testing mode
            break
        # Take max 10 patches per WSI in testing mode
        limited_entries = entries[:10]
        limited_wsi_groups[file_id] = limited_entries
        print(f"  Testing mode: {file_id} limited to {len(limited_entries)} patches")
    
    wsi_groups = limited_wsi_groups

# Create output directory for individual metadata files
metadata_dir = Path(snakemake.output.metadata_per_wsi_dir)
metadata_dir.mkdir(parents=True, exist_ok=True)

# Write individual JSON files for each WSI
individual_files_written = 0
total_patches_written = 0
wsi_summary = {}

for file_id, entries in wsi_groups.items():
    # Apply max patches per WSI limit (non-testing mode)
    if not snakemake.params.testing_mode and len(entries) > snakemake.params.max_patches_per_wsi:
        entries = entries[:snakemake.params.max_patches_per_wsi]
        print(f"Limited {file_id} to {snakemake.params.max_patches_per_wsi} patches")
    
    # Write individual JSON file for this WSI
    output_file = metadata_dir / f"{file_id}.json"
    with open(output_file, "w") as f:
        json.dump(entries, f, indent=2)
    
    # Update summary statistics
    wsi_id = entries[0].get("wsi_id", "unknown") if entries else "unknown"
    wsi_summary[file_id] = {"wsi_id": wsi_id, "count": len(entries)}
    individual_files_written += 1
    total_patches_written += len(entries)
    
    print(f"  Created {output_file} with {len(entries)} patches")

print(f"Created {individual_files_written} individual metadata files")
print(f"Total patches across all files: {total_patches_written}")

# Also create the legacy combined filtered metadata for backward compatibility
all_filtered_entries = []
for entries in wsi_groups.values():
    all_filtered_entries.extend(entries)

with open(snakemake.output.metadata_filtered, "w") as f:
    json.dump(all_filtered_entries, f, indent=2)

print(f"Created legacy combined metadata file: {snakemake.output.metadata_filtered}")

# Write summary report
with open(snakemake.output.metadata_summary, "w") as f:
    f.write("PathGen Metadata Filtering Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Original entries: {len(data)}\n")
    f.write(f"Available WSI files: {len(available_files)}\n")
    f.write(f"Filtered entries: {len(all_filtered_entries)}\n")
    f.write(f"Unique WSIs processed: {len(wsi_groups)}\n")
    f.write(f"Individual metadata files created: {individual_files_written}\n")
    f.write(f"Testing mode: {'Yes' if snakemake.params.testing_mode else 'No'}\n")
    f.write("\nPer-WSI Summary:\n")
    f.write("-" * 30 + "\n")
    
    for file_id, info in wsi_summary.items():
        f.write(f"{file_id} ({info['wsi_id']}): {info['count']} patches\n")

print(f"Summary report saved to: {snakemake.output.metadata_summary}")
print("Per-WSI metadata filtering completed successfully!")