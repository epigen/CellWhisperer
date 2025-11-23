import json
import pandas as pd
from pathlib import Path

print("Filtering PathGen metadata based on available WSI files...")

# Load metadata JSON
with open(snakemake.input.metadata_json, "r") as f:
    data = json.load(f)
print(f"Loaded {len(data)} entries from metadata JSON")


# Find available WSI files
wsi_dir = Path(snakemake.params.wsi_dir)
available_files = set(f.stem for f in wsi_dir.glob("*.svs"))
print(f"Found {len(available_files)} available WSI files")

# Filter metadata based on available WSI files
filtered_data = []
for entry in data:
    file_id = entry.get("file_id", "")
    if file_id and file_id in available_files:
        filtered_data.append(entry)

print(f"Filtered to {len(filtered_data)} entries with available WSI files")

# In testing mode, further limit the data
if snakemake.params.testing_mode:
    print("TESTING MODE: Further limiting data")

    # Group by file_id and take only first 2 WSIs with limited patches
    file_id_groups = {}
    for entry in filtered_data:
        file_id = entry["file_id"]
        if file_id not in file_id_groups:
            file_id_groups[file_id] = []
        file_id_groups[file_id].append(entry)

    # Take first 2 file_ids and limit patches per WSI
    test_data = []
    for i, (file_id, entries) in enumerate(file_id_groups.items()):
        if i >= 2:  # Only first 2 WSIs
            break
        # Take max 10 patches per WSI in testing mode
        test_entries = entries[:10]
        test_data.extend(test_entries)

    filtered_data = test_data
    print(f"Testing mode: limited to {len(filtered_data)} entries from 2 WSIs")

# Group by WSI for summary
wsi_summary = {}
for entry in filtered_data:
    file_id = entry["file_id"]
    wsi_id = entry.get("wsi_id", "unknown")
    if file_id not in wsi_summary:
        wsi_summary[file_id] = {"wsi_id": wsi_id, "count": 0}
    wsi_summary[file_id]["count"] += 1

print(f"Final dataset summary:")
for file_id, info in wsi_summary.items():
    print(f"  {file_id} ({info['wsi_id']}): {info['count']} patches")

# Save filtered metadata
with open(snakemake.output.metadata_filtered, "w") as f:
    json.dump(filtered_data, f, indent=2)

print(f"Filtered metadata saved to: {snakemake.output.metadata_filtered}")
print(f"Total entries: {len(filtered_data)}")
print(f"Unique WSIs: {len(wsi_summary)}")
