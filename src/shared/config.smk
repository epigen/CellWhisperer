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
    gpu_size="medium",
    num_gpus=1,
    num_cpus=5,
    cluster_name=config["cluster_name"],
    time="08:00:00",
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

        gres = "gres=gpu:{gpu_type}:{num_gpus}".format(
            gpu_type=gpu_type, num_gpus=num_gpus
        )

    elif cluster_name == "sherlock":
        # gpu_type = {
        #     "small": "GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB",
        #     "medium": "GPU_MEM:48GB",
        #     "large": "GPU_MEM:80GB",
        # }[gpu_size]
        if True or gpu_size in ["large", "medium"]:
            partition = "cmackall"
            gpu_type = "GPU_CC:8.9|GPU_CC:9.0|GPU_GEN:HPR"  # bfloat16 support
        else:
            gpu_type = (
                "GPU_SKU:L40S"  # bfloat16 support  # GPU_CC:8.9|GPU_CC:9.0|GPU_GEN:HPR|
            )
        gres = "gpus={num_gpus} constraint={constraint}".format(
            num_gpus=num_gpus, constraint=gpu_type
        )
        qos = ""
    elif cluster_name == "ilc":
        pass  # TODO implement blackwell1 and hyperturing2
    else:
        gpu_type = {"small": "3g.20gb", "medium": "a100", "large": "a100-sxm4-80gb"}[
            gpu_size
        ]
        qos = "qos={gpu_type}".format(gpu_type=gpu_type)
        gres = "gres=gpu:{gpu_type}:{num_gpus}".format(
            gpu_type=gpu_type, num_gpus=num_gpus
        )

    return "cpus-per-task={num_cpus} {gres} {qos} partition={partition} time={time}".format(
        num_cpus=num_cpus, gres=gres, qos=qos, partition=partition, time=time
    )


HTTP = HTTPRemoteProvider()
