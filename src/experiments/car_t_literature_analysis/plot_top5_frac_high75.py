import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

STATS_PATH = "results/patient_signals/stats_mannwhitney.csv"
OUT_PATH = "results/patient_signals/bar_top5_frac_high75.png"

# Load and filter
df = pd.read_csv(STATS_PATH)
df = df[df["agg"] == "frac_high75"].copy()

# Top 5 by smallest p-value
df = df.sort_values("pvalue").head(5)

# Plot values with direction
vals = -np.log10(df["pvalue"].values)
terms = df["term"].values
alpha = 0.05
colors = ["tab:red" if p < alpha else "gray" for p in df["pvalue"].values]
# Direction by OR - NR
delta = df["OR_mean"].values - df["NR_mean"].values
vals_signed = vals * np.sign(delta)

# Format ratio labels and invert order
def _fmt_term(t):
    if t.startswith("ratio_frac_high75_"):
        t = t.replace("ratio_frac_high75_", "")
        t = t.replace("__", "/")
    return t

terms_fmt = [_fmt_term(t) for t in terms]
vals_inv = vals_signed[::-1]
terms_inv = terms_fmt[::-1]
colors_inv = colors[::-1]

# Small figure; horizontal bars
plt.figure(figsize=(1.4, 1.425))
plt.barh(terms_inv, vals_inv, color=colors_inv)
plt.axvline(0, color="black", linewidth=0.6)
plt.xlabel("signed -log10(p); positive=OR>NR")
plt.ylabel("Term (ratios shown as A/B)")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
plt.savefig(OUT_PATH.replace(".png", ".svg"))
plt.close()

print(f"Saved: {OUT_PATH}")
