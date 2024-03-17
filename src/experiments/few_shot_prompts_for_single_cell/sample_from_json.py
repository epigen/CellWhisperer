import json
import numpy as np
import pandas as pd
import random


fn = "/mnt/muwhpc/cellwhisperer/results/cellxgene_census/annotations.json"
fn = "/msc/home/mschae83/cellwhisperer/results/cellxgene_census/annotations.json"

with open(fn, "r") as f:
    data = json.load(f)

# sample 10 random keys

sample_keys = random.sample(list(data.keys()), 10)

# write the contents to individual files
for key in sample_keys:
    with open(f"{key}.json", "w") as f:
        json.dump(data[key], f)
