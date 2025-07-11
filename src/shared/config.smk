import subprocess
from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
from pathlib import Path

PROJECT_DIR = Path(
    subprocess.check_output("git rev-parse --show-toplevel", shell=True)
    .decode("utf-8")
    .strip()
)
configfile: PROJECT_DIR / "config.yaml"


CW_CLIP_MODELS = [
    model_path
    for model_name, model_path in config["model_name_path_map"].items()
    if "cellwhisperer" in model_name
]
CLIP_MODEL = config["model_name_path_map"]["cellwhisperer"]
LLAVA_BASE_MODEL = config["model_name_path_map"]["llava_base_llm"]


def slurm_gres(
    gpu_size="medium", num_gpus=1, num_cpus=5, cluster_name=config["cluster_name"]
):
    """
    gpu_size: small == 20gb, medium == 40gb, large == 80gb
    """

    partition = "gpu"
    qos = "qos=gpu"
    if cluster_name == "lustre":
        gpu_type = {
            "small": "l4_gpu",
            "medium": "h100hgx",  # alternative h100_4g.47gb
            "large": "h100pcie",
        }[gpu_size]

        gres = f"gres=gpu:{gpu_type}:{num_gpus}"

    elif cluster_name == "sherlock":
        gpu_type = {
            "small": "GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB",
            "medium": "GPU_MEM:48GB",
            "large": "GPU_MEM:80GB",
        }[gpu_size]
        gres = f'gpus={num_gpus} constraint="{gpu_type}"'
        qos = ""
    else:
        gpu_type = {"small": "3g.20gb", "medium": "a100", "large": "a100-sxm4-80gb"}[
            gpu_size
        ]
        qos = f"qos={gpu_type}"
        gres = f"gres=gpu:{gpu_type}:{num_gpus}"

    return f"cpus-per-task={num_cpus} {gres} {qos} partition={partition}"


HTTP = HTTPRemoteProvider()
