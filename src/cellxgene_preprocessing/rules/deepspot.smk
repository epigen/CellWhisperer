import os
from pathlib import Path
DEESPOT_MODELS_DIR = PROJECT_DIR / config["model_name_path_map"]["deepspot_pretrained"]

# Rule to process images for deepseek
rule deepspot_inference:
    input:
        Colon_HEST1K_pretrained = DEESPOT_MODELS_DIR / "Colon_HEST1K/final_model.pkl",
        Kidney_HEST1K_pretrained = DEESPOT_MODELS_DIR /  "Kidney_HEST1K/final_model.pkl",
        Kidney_Lung_USZ_pretrained = DEESPOT_MODELS_DIR / "Kidney_Lung_USZ/final_model.pkl",
        Melanoma_TuPro_pretrained = DEESPOT_MODELS_DIR / "Melanoma_TuPro/final_model.pkl",
        uni = PROJECT_DIR / config["model_name_path_map"]["uni"] / "pytorch_model.bin",
        hoptimus0 = PROJECT_DIR / config["model_name_path_map"]["hoptimus0"] / "pytorch_model.bin",
        input_image = PROJECT_DIR / "resources/{dataset}/image.jpg",
        image_config = PROJECT_DIR / "resources/{dataset}/config.json",
    output:
        read_count_table = PROJECT_DIR / config["paths"]["read_count_table"]
    conda:
        "deepspot"
    resources:
        mem_mb=240000,
        slurm="-q a100-sxm4-80gb -c 16 --partition gpu --gres=gpu:a100-sxm4-80gb:1"
    log:
        notebook="../log/deepspot_inference_{dataset}.py.ipynb"
    notebook:
        "../notebooks/deepspot_inference.py.ipynb"