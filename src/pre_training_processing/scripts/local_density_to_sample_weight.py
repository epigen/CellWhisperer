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


data = np.load(snakemake.input.representation, allow_pickle=True)


# https://github.com/lmcinnes/umap/issues/763#issuecomment-1520913313
# class ProgressWriter:
#     def write(self, text):

#         match = re.search(r"(\d+)/(\d+)", text)
#         if match:
#             n, total = map(int, match.groups())
#             print("custom progress", n, total)
#             # custom reporting logic here

#     def flush(self):
#         pass

embedding, r_orig, r_emb = umap.UMAP(
    densmap=True,
    dens_lambda=2.0,
    n_neighbors=30,
    output_dens=True,
    tqdm_kwds={"disable": False},  # "file": progress_writer"
    verbose=True,
).fit_transform(data["representation"])

# normalize such that the sum of radii corresponds to len(dataset)
# note that the radii are in log space (so if we want greater variation, take their exp)

weight = r_orig

# Clamp outliers (based on 2 stds). They would otherwise explode a bit. NOTE: could be improved with soft clamping
mean = np.mean(weight)
std_dev = np.std(weight)
weight = np.clip(weight, mean - 2 * std_dev, mean + 2 * std_dev)

# Revert log, then square to get from radius to area (yields satisfying distribution)
weight = np.exp(weight) ** 2

# Normalize to retain library size (total weight = len(dataset))
weight /= weight.mean()

np.savez(
    snakemake.output.weight,
    weight=weight,
    orig_ids=data["orig_ids"],
)

# plot as well
plt.subplots(figsize=(3, 2))
sns.histplot(weight)
# plt.xlim(0, 3)
plt.savefig(snakemake.output.plot_orig_radii, bbox_inches="tight")

print(
    "Please verify the distribution plot. We want many values at 1 and slightly below it (e.g. 0.8). And some values at ~2-3."
)
