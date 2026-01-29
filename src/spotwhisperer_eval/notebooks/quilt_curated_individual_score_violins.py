from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind

# Inputs from Snakemake
inputs = list(snakemake.input.individual_scores)
out_plot = Path(snakemake.output.plot)
out_stats = Path(snakemake.output.stats)
labels = snakemake.params.test_datasets

out_plot.parent.mkdir(parents=True, exist_ok=True)
out_stats.parent.mkdir(parents=True, exist_ok=True)

# Load and annotate
dfs = []
for csv, label in zip(inputs, labels):
    df = pd.read_csv(csv)
    df["dataset"] = label
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

# Per-sample mean of both directions
df["clip_score_mean"] = (df["clip_score_left_right"] + df["clip_score_right_left"]) / 2.0

# Style and plot
plt.style.use(str(snakemake.input.mpl_style))
fig, ax = plt.subplots(figsize=(2.5, 1))
sns.violinplot(data=df, y="dataset", x="clip_score_mean", inner="quartile", cut=0, ax=ax)
ax.set_title("Per-sample mean CLIP score (quilt vs curated)")
ax.set_xlabel("")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(out_plot, dpi=300, bbox_inches="tight")

# Statistical comparison (unpaired)
x = df.loc[df.dataset == "quilt1m", "clip_score_mean"].to_numpy()
y = df.loc[df.dataset == "quilt1m_curated", "clip_score_mean"].to_numpy()
u_two, p_two = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
u_greater, _ = mannwhitneyu(x, y, alternative="greater", method="asymptotic")
n1, n2 = len(x), len(y)
auc = u_greater / (n1 * n2)
cliffs_delta = 2 * auc - 1
t_stat, p_welch = ttest_ind(x, y, equal_var=False)
rng = np.random.default_rng(0)
B = 2000
diffs = np.empty(B)
for b in range(B):
    xb = x[rng.integers(0, n1, n1)]
    yb = y[rng.integers(0, n2, n2)]
    diffs[b] = xb.mean() - yb.mean()
ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

with open(out_stats, "w") as f:
    f.write("Two-sided Mann-Whitney U p-value: {:.6g}\n".format(p_two))
    f.write("Effect size (AUC): {:.6f}\n".format(auc))
    f.write("Cliff's delta: {:.6f}\n".format(cliffs_delta))
    f.write("Welch's t-test p-value: {:.6g}\n".format(p_welch))
    f.write(
        "Bootstrap 95% CI (mean difference quilt1m - curated): [{:.6f}, {:.6f}]\n".format(
            ci_low, ci_high
        )
    )
