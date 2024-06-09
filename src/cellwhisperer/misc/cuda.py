import os
import torch
import numpy as np


def get_device() -> torch.device:
    """
    Sets the GPU with the most free memory as device, if available.

    NOTE: This function is only relevant to Peter at the moment, who works on a machine with GPU sharing
    NOTE: the use of this function might interfere with pytorch lightning, which itself has mechanisms to choose and use CPUs
    """
    if torch.cuda.is_available():
        if os.environ.get("HOSTNAME") == "011sv149":
            os.system(
                "nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp_gpumem.txt"
            )
            memory_available = [
                int(x.split()[2]) for x in open("tmp_gpumem.txt", "r").readlines()
            ]
            os.system("rm tmp_gpumem.txt")
            device = torch.device(int(str(np.argmax(memory_available))))
            print("Using GPU: {}".format(device))
        else:
            device = torch.device("cuda")
            print("Using GPU: {}".format(device))
    else:
        print("No GPU available. Falling back to CPU")
        device = torch.device("cpu")

    return device
