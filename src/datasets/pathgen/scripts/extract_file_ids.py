import json
import sys
from pathlib import Path
from collections import Counter

print("Extracting file IDs from PathGen metadata...")

# Load metadata JSON
with open(snakemake.input.metadata_json, "r") as f:
    data = json.load(f)
print(f"Loaded {len(data)} entries from metadata JSON")

with open(snakemake.input.broken_ids, "r") as f:
    broken_ids = set(line.strip() for line in f)

# Extract file IDs and group by WSI
file_id_to_wsi = {}
wsi_to_entries = {}

for entry in data:
    file_id = entry.get("file_id", "")
    wsi_id = entry.get("wsi_id", "")
    if file_id in broken_ids:
        continue

    if file_id and wsi_id:
        file_id_to_wsi[file_id] = wsi_id
        if file_id not in wsi_to_entries:
            wsi_to_entries[file_id] = []
        wsi_to_entries[file_id].append(entry)

print(f"Found {len(file_id_to_wsi)} unique WSI files")
print(f"Total entries: {sum(len(entries) for entries in wsi_to_entries.values())}")

# Handle empty metadata case
if not file_id_to_wsi and snakemake.params.testing_mode:
    print("TESTING MODE: No file IDs found in metadata, creating sample file IDs")
    # Use real GDC file IDs that are publicly accessible for testing
    sample_file_ids = [
        "35bc77a6-af3f-44e4-abb9-6ed1932bd3e4",  # Sample TCGA file ID
        "d6232722-c4e2-471f-982d-b7074085ce68",  # Another sample TCGA file ID
    ]
    file_ids_to_use = sample_file_ids
    # Create dummy wsi mapping for summary
    for file_id in sample_file_ids:
        file_id_to_wsi[file_id] = f"SAMPLE-WSI-{file_id[:8]}"
        wsi_to_entries[file_id] = [
            {"file_id": file_id, "wsi_id": file_id_to_wsi[file_id]}
        ]
elif snakemake.params.testing_mode:
    print("TESTING MODE: Limiting to first 2 WSIs")
    file_ids_to_use = list(file_id_to_wsi.keys())[:2]
else:
    file_ids_to_use = list(file_id_to_wsi.keys())

print(f"Selected {len(file_ids_to_use)} WSI files for processing")

# Write file IDs to output
with open(snakemake.output.file_ids_list, "w") as f:
    for file_id in file_ids_to_use:
        f.write(f"{file_id}\n")

# Create metadata summary
total_patches = sum(len(wsi_to_entries[file_id]) for file_id in file_ids_to_use)
with open(snakemake.output.metadata_summary, "w") as f:
    f.write(f"PathGen-1.6M Metadata Summary\n")
    f.write(f"================================\n")
    f.write(f"Total entries in JSON: {len(data)}\n")
    f.write(f"Unique WSI files: {len(file_id_to_wsi)}\n")
    f.write(f"Selected WSI files: {len(file_ids_to_use)}\n")
    f.write(f"Selected patches: {total_patches}\n")
    f.write(f"Testing mode: {snakemake.params.testing_mode}\n")
    f.write(f"\nSelected File IDs:\n")
    for file_id in file_ids_to_use:
        wsi_id = file_id_to_wsi[file_id]
        num_patches = len(wsi_to_entries[file_id])
        f.write(f"  {file_id} -> {wsi_id} ({num_patches} patches)\n")

print(f"File IDs written to: {snakemake.output.file_ids_list}")
print(f"Summary written to: {snakemake.output.metadata_summary}")
