import shutil
import os

# extracted from https://huggingface.co/datasets/MahmoodLab/hest/raw/main/HEST_v1_1_0.csv with gemini, asking for crc
crc_ids = [
    "TENX156",
    "TENX155",
    "TENX154",
    "TENX152",
    "TENX149",
    "TENX148",
    "TENX147",
    "MISC73",
    "MISC72",
    "MISC71",
    "MISC70",
    "MISC69",
    "MISC68",
    "MISC67",
    "MISC66",
    "MISC65",
    "MISC64",
    "MISC63",
    "MISC62",
    "MISC58",
    "MISC57",
    "MISC56",
    "MISC51",
    "MISC50",
    "MISC49",
    "MISC48",
    "MISC47",
    "MISC46",
    "MISC45",
    "MISC44",
    "MISC43",
    "MISC42",
    "MISC41",
    "MISC40",
    "MISC39",
    "MISC38",
    "MISC37",
    "MISC36",
    "MISC35",
    "MISC34",
    "MISC33",
    "TENX139",
    "TENX128",
    "TENX111",
    "TENX92",
    "TENX91",
    "TENX90",
    "TENX89",
    "TENX70",
    "TENX49",
    "TENX29",
    "TENX28",
    "ZEN49",
    "ZEN48",
    "ZEN47",
    "ZEN46",
    "ZEN45",
    "ZEN44",
    "ZEN43",
    "ZEN42",
    "ZEN40",
    "ZEN39",
    "ZEN38",
    "ZEN36",
]

for crc_id in crc_ids:
    # ln -s to absolute path /oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/hest1k/h5ads/full_data_{crc_id}.h5ad
    target_path = f"/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/hest1k/h5ads/full_data_{crc_id}.h5ad"
    if os.path.exists(target_path):
        link_name = f"/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/hest1k_crc/h5ads/full_data_{crc_id}.h5ad"
        if not os.path.exists(link_name):
            os.symlink(target_path, link_name)
            print(f"Created symlink for {crc_id}")
        else:
            print(f"Symlink already exists for {crc_id}")
    else:
        print(f"Target path does not exist for {crc_id}: {target_path}")
