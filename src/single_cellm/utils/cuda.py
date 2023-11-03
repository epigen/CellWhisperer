import os
import torch
import numpy as np


def set_freest_gpu_as_device() -> int:
    """
    Sets the GPU with the most free memory as device, if available.
    """
    if torch.cuda.is_available():
        os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp_gpumem.txt")
        memory_available = [
            int(x.split()[2]) for x in open("tmp_gpumem.txt", "r").readlines()
        ]
        os.system("rm tmp_gpumem.txt")
        device = torch.device(int(str(np.argmax(memory_available))))
        torch.cuda.set_device(device)
        print("Using GPU: {}".format(device))
    else:
        print("No GPU available. Skipping setting freest GPU as device...")
