import warnings

warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
import torch
import os

from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.utils.inference import score_left_vs_right

# Inputs
H5AD_PATH = "cellxgene_B_Product_lowburden.light.h5ad"
METADATA_XLSX = "CD19_atlas_patient_metadata-KT.xlsx"  # optional; not used if obs has required fields
OUT_DIR = "results/patient_signals"

# Queries
QUERY_GROUPS = {
    "T cell subsets": [
        "CD8+ T cells",
        "CD4+ T cells",
        "Memory T cells",
        "Naive T cells",
        "Effector T cells",
        "Regulatory T cells",
        "Senescent T cells",
    ],
    "Functional states": [
        "Activated T cells",
        "Anergic cells",
        "Cytotoxic T cells",
        "Th1 cells",
        "Th17 cells",
    ],
    "Cell cycle & survival": [
        "Proliferating cells",
        "Apoptotic cells",
        "Quiescent cells",
    ],
    "Manufacturing/engineering": [
        "Transduced cells",
        "CAR-expressing cells",
    ],
    "Stress/dysfunction": [
        "Hypoxic cells",
        "Damaged cells",
    ],
    "Metabolic states": [
        "Glycolytic cells",
        "Oxidative cells",
    ],
    "Differentiation": [
        "Stem cell-like T cells",
        "Terminally differentiated cells",
    ],
    "Antigen experience": [
        "Antigen-experienced cells",
        "Antigen-naive cells",
    ],
}
ALL_TERMS = [t for ts in QUERY_GROUPS.values() for t in ts]

os.makedirs(OUT_DIR, exist_ok=True)

# Load data
adata = ad.read_h5ad(H5AD_PATH)

# Subset file already filtered to B_Product and low-burden (+ missing-burden included)
# No additional filtering here

# Harvest needed arrays and release AnnData
obs = adata.obs
idx = adata.obs_names.copy()
patient_ids = obs["patient_id"].values
responses = obs["Response_3m"].values
import gc

# Load model for text embedding and logit scale
CKPT_PATH = "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/models/jointemb/old_format/spotwhisperer_cellxgene_census__archs4_geo.ckpt"
pl_model, tokenizer, transcriptome_processor, image_processor = (
    load_cellwhisperer_model(model_path=CKPT_PATH, eval=True)
)
model = pl_model.model
logit_scale = model.discriminator.temperature.exp()


# Score transcripts vs texts: returns n_right * n_left = n_terms * n_cells
scores_terms_vs_cells, _ = score_left_vs_right(
    left_input=adata,
    right_input=ALL_TERMS,
    logit_scale=logit_scale,
    model=model,
    average_mode=None,
    grouping_keys=None,
    batch_size=32,
    score_norm_method=None,
)

# Convert to DataFrame: n_cells x n_terms
scores_df = pd.DataFrame(
    scores_terms_vs_cells.T.cpu().numpy(), index=idx, columns=ALL_TERMS
)

# Attach patient_id and Response_3m
scores_df["patient_id"] = patient_ids
scores_df["Response_3m"] = responses
scores_df = scores_df[scores_df["Response_3m"].isin(["OR", "NR"])].copy()
scores_df.to_csv(os.path.join(OUT_DIR, "cell_level_scores.csv"))

# Aggregations per patient
agg_mean = scores_df.groupby("patient_id")[ALL_TERMS].mean()
# define a threshold per term as the 75th percentile across all cells for that term
thresholds = scores_df[ALL_TERMS].quantile(0.75)
agg_frac_high = scores_df.groupby("patient_id")[ALL_TERMS].apply(
    lambda g: (g > thresholds).mean()
)
agg_max = scores_df.groupby("patient_id")[ALL_TERMS].max()
# 85th percentile per patient-term
agg_p85 = scores_df.groupby("patient_id")[ALL_TERMS].quantile(0.85)

# Merge with response
patient_response = scores_df.groupby("patient_id")["Response_3m"].first()
agg_mean["Response_3m"] = patient_response
agg_frac_high["Response_3m"] = patient_response
agg_max["Response_3m"] = patient_response
agg_p85["Response_3m"] = patient_response

# Define ratio pairs and add ratios per aggregation
RATIO_PAIRS = [
    ("CD8+ T cells", "CD4+ T cells"),
    ("Memory T cells", "Naive T cells"),
    ("Effector T cells", "Regulatory T cells"),
    ("Activated T cells", "Anergic cells"),
    ("Cytotoxic T cells", "Anergic cells"),
    ("Th1 cells", "Th17 cells"),
    ("Proliferating cells", "Quiescent cells"),
    ("CAR-expressing cells", "Transduced cells"),
    ("Glycolytic cells", "Oxidative cells"),
    ("Stem cell-like T cells", "Terminally differentiated cells"),
    ("Antigen-experienced cells", "Antigen-naive cells"),
]


def add_ratios(df, agg_name):
    ratio_cols = []
    for a, b in RATIO_PAIRS:
        if a in df.columns and b in df.columns:
            col = f"ratio_{agg_name}_{a}__{b}"
            df[col] = df[a] / (df[b] + 1e-6)
            ratio_cols.append(col)
    return ratio_cols


ratio_cols_mean = add_ratios(agg_mean, "mean")
ratio_cols_frac = add_ratios(agg_frac_high, "frac_high75")
ratio_cols_max = add_ratios(agg_max, "max")
ratio_cols_p85 = add_ratios(agg_p85, "p85")

# Save tables
agg_mean.to_csv(os.path.join(OUT_DIR, "patient_agg_mean.csv"))
agg_frac_high.to_csv(os.path.join(OUT_DIR, "patient_agg_frac_high75.csv"))
agg_max.to_csv(os.path.join(OUT_DIR, "patient_agg_max.csv"))
agg_p85.to_csv(os.path.join(OUT_DIR, "patient_agg_p85.csv"))
agg_mean[ratio_cols_mean].to_csv(os.path.join(OUT_DIR, "patient_agg_ratios_mean.csv"))
agg_frac_high[ratio_cols_frac].to_csv(
    os.path.join(OUT_DIR, "patient_agg_ratios_frac_high75.csv")
)
agg_max[ratio_cols_max].to_csv(os.path.join(OUT_DIR, "patient_agg_ratios_max.csv"))
agg_p85[ratio_cols_p85].to_csv(os.path.join(OUT_DIR, "patient_agg_ratios_p85.csv"))

# Statistical testing (OR vs NR)
from scipy.stats import mannwhitneyu

results = []
for agg_name, df, extra_cols in [
    ("mean", agg_mean, ratio_cols_mean),
    ("frac_high75", agg_frac_high, ratio_cols_frac),
    ("max", agg_max, ratio_cols_max),
    ("p85", agg_p85, ratio_cols_p85),
]:
    # test base terms
    for term in ALL_TERMS:
        or_vals = df[df["Response_3m"] == "OR"][term]
        nr_vals = df[df["Response_3m"] == "NR"][term]
        if len(or_vals) >= 3 and len(nr_vals) >= 3:
            stat, p = mannwhitneyu(or_vals, nr_vals, alternative="two-sided")
            results.append(
                {
                    "agg": agg_name,
                    "term": term,
                    "pvalue": p,
                    "OR_mean": or_vals.mean(),
                    "NR_mean": nr_vals.mean(),
                }
            )
    # test ratios
    for col in extra_cols:
        or_vals = df[df["Response_3m"] == "OR"][col]
        nr_vals = df[df["Response_3m"] == "NR"][col]
        if len(or_vals) >= 3 and len(nr_vals) >= 3:
            stat, p = mannwhitneyu(or_vals, nr_vals, alternative="two-sided")
            results.append(
                {
                    "agg": agg_name,
                    "term": col,
                    "pvalue": p,
                    "OR_mean": or_vals.mean(),
                    "NR_mean": nr_vals.mean(),
                }
            )

res_df = pd.DataFrame(results).sort_values("pvalue")
res_df.to_csv(os.path.join(OUT_DIR, "stats_mannwhitney.csv"), index=False)

# Violin plots for significant terms
import matplotlib.pyplot as plt
import seaborn as sns

alpha = 0.05
sig = res_df[res_df["pvalue"] < alpha]
sig_terms = sig.groupby("agg")["term"].unique()


def plot_violin(df, agg_name, term):
    plot_df = df[[term, "Response_3m"]].copy()
    plot_df = plot_df[plot_df["Response_3m"].isin(["OR", "NR"])]
    plot_df["patient_id"] = plot_df.index

    plt.figure(figsize=(4.0, 3.2))
    sns.violinplot(data=plot_df, x="Response_3m", y=term, inner=None, cut=0)
    sns.stripplot(
        data=plot_df, x="Response_3m", y=term, color="black", size=4, jitter=True
    )
    plt.title(f"{agg_name}: {term}")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"violin_{agg_name}_{term.replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=200)
    plt.savefig(out_path.replace(".png", ".svg"))
    plt.close()


for agg_name, terms in sig_terms.items():
    df = {
        "mean": agg_mean,
        "frac_high75": agg_frac_high,
        "max": agg_max,
        "p85": agg_p85,
    }[agg_name]
    for term in terms:
        plot_violin(df, agg_name, term)

print(f"Done. Saved outputs to {OUT_DIR}")
