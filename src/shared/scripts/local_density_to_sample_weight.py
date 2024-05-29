"""
Use the local radii (inverse measure of local density defined in the densMAP paper, log-transformed)
to compute a sample weight. The weight is a measure of how much the sample should contribute to the
loss function. It is computed as the area of the local density (in log space) and normalized such that
the sum of all weights equals the number of samples.
"""

import numpy as np
import umap
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import re
import sys
import logging


data = np.load(snakemake.input.representation, allow_pickle=True)

# Code to configure snakemake logging to redirect all std out to the logfile
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=snakemake.log[0],
    filemode="w",
)


# https://github.com/lmcinnes/umap/issues/763#issuecomment-1520913313
class ProgressWriter:
    def write(self, text):
        match = re.search(r"(\d+)/(\d+)", text)
        if match:
            n, total = map(int, match.groups())
            # custom reporting logic here
            logging.info(f"Progress: {n}/{total}")
        else:
            logging.info(text)

    def flush(self):
        pass


logging.info("Starting local density to sample weight conversion")
embedding, r_orig, r_emb = umap.UMAP(
    densmap=True,
    dens_lambda=2.0,
    n_neighbors=30,
    output_dens=True,
    tqdm_kwds={"disable": False, "file": ProgressWriter()},
    verbose=True,
).fit_transform(data["representation"])
logging.info("Finished UMAP")

# normalize such that the sum of radii corresponds to len(dataset)
# note that the radii are in log space (so if we want greater variation, take their exp)

np.savez(snakemake.output.orig_radii, r_orig)  # type: ignore [reportUndefinedVariable]

exped = np.exp(r_orig)

exped_shifted = exped + (
    exped.mean() / 10
)  # shift the distribution a bit to the right to avoid close-to-zero values

clipped = np.clip(exped_shifted, 0, np.percentile(exped_shifted, 99.9))
weight = clipped / np.mean(clipped)

np.savez(
    snakemake.output.weight,  # type: ignore [reportUndefinedVariable]
    weight=weight,
    orig_ids=data["orig_ids"],
)

# plot as well
plt.subplots(figsize=(3, 2))
sns.histplot(weight)
# plt.xlim(0, 3)
plt.savefig(snakemake.output.plot_orig_radii, bbox_inches="tight")  # type: ignore [reportUndefinedVariable]

print(
    "Please verify the distribution plot. We want many values at 1 and slightly below it (e.g. 0.8). And some values at ~2-3."
)
