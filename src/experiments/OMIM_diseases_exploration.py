import anndata
import pandas as pd
from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.utils.inference import score_transcriptomes_vs_texts
import torch
from pathlib import Path
from cellwhisperer.config import get_path
import scanpy as sc

sc.set_figure_params(vector_friendly=True, dpi_save=200)
import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib

sc.set_figure_params(
    vector_friendly=True, dpi_save=300
)  # Makes PDFs of scatter plots much smaller in size but still high-quality


def plot_term_score_umaps(adata, term, scores_this_term, df_stack, outdir):
    top_5_cluster_labels = df_stack[df_stack["term_without_prefix"] == term][
        "cluster_label"
    ].values[:5]
    label_score_dict = {
        x: round(float(y), 1)
        for x, y in zip(
            top_5_cluster_labels,
            df_stack[df_stack["term_without_prefix"] == term]["logits"].values[:5],
        )
    }

    adata.obs[f"per_transcriptome_score_for_{term}"] = scores_this_term
    adata.obs["cluster_label_top_5"] = [
        f"{x}\nScore_cluster:{round(label_score_dict[x],1)}"
        if x in top_5_cluster_labels
        else "other"
        for x in adata.obs["cluster_label"]
    ]
    order = [
        f"{x}\nScore_cluster:{round(label_score_dict[x],1)}"
        for x in top_5_cluster_labels
    ]

    order.append("other")

    adata.obs["cluster_label_top_5"] = pd.Categorical(
        adata.obs["cluster_label_top_5"], categories=order, ordered=True
    )

    langauge_annots_top5 = (
        adata.obs.sort_values(by=f"per_transcriptome_score_for_{term}", ascending=False)
        .head(5)["natural_language_annotation"]
        .tolist()
    )

    # plot
    best_score = adata.obs[f"per_transcriptome_score_for_{term}"].max()
    fig, axes = plt.subplots(
        2, 2, figsize=(15, 7), gridspec_kw={"height_ratios": [1, 0.3]}
    )

    # Score per transcriptome
    sc.pl.embedding(
        adata,
        basis="X_umap",
        color=f"per_transcriptome_score_for_{term}",
        cmap="RdBu_r",
        vmin=-best_score,
        vmax=best_score,
        show=False,
        ax=axes[0][0],
    )

    # Cluster labels
    sc.pl.embedding(
        adata[adata.obs["cluster_label_top_5"] == "other"],
        basis="X_umap",
        color=["cluster_label_top_5"],
        show=False,
        ax=axes[0][1],
        palette={"other": "silver"},
        s=0.5,
    )
    axes[0][1].get_legend().remove()
    sc.pl.embedding(
        adata[adata.obs["cluster_label_top_5"] != "other"],
        basis="X_umap",
        color=["cluster_label_top_5"],
        show=False,
        ax=axes[0][1],
        legend_loc="on data",
        legend_fontsize=6,
        s=0.5,
    )

    plt.sca(axes[1][0])
    plt.text(
        0.5,
        0.5,
        "Examples from top cluster: \n" + ("\n".join(langauge_annots_top5)),
        horizontalalignment="left",
        verticalalignment="center",
        transform=axes[1][0].transAxes,
        fontsize=9,
    )

    # Cosmetics
    for ax in plt.gcf().axes:
        ax.set_title("")
        ax.set_facecolor("white")
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes[1]:
        ax.axis("off")
    plt.subplots_adjust(right=0.7)
    plt.suptitle(f"Top clusters for: {term}")
    cbar = axes[0][0].collections[0].colorbar
    cbar.set_label("Score for disease keyword", fontsize=11)

    # Save
    plt.savefig(outdir / f"umap_{term.replace('/','_')}.png", dpi=660)
    plt.savefig(outdir / f"umap_{term.replace('/','_')}.pdf")
    plt.close()


### PARAMETERS ###
library = "OMIM_Expanded"
BASEDIR = "/home/peter/peter_on_isilon/cellwhisperer/single-cellm/results/"
ckpt_file_name = "cellwhisperer_clip_v1_epoch5_maxval.ckpt"
max_terms_per_cluster = 3
max_terms_total = 20
compare_top_vs_clusternumber = 2  # note: 0-indexed
score_norm_method = None
add_sum_of_all_scores = True

### IMPORT / EXPORT ###
df = pd.read_csv(f"{BASEDIR}/1b/cellwhisperer_annotated_clusters.csv")
adata = anndata.read_h5ad(f"{BASEDIR}/1b/llava_annotated_clusters.h5ad")
language_labels = pd.read_csv(f"{BASEDIR}/1b/gpt4_labels.csv", index_col=0)[
    "GPT4_generated_labels"
]
npz = np.load(f"{BASEDIR}/1b/full_output.npz")
annotations_per_transcriptome = json.load(
    open(f"{BASEDIR}/archs4_geo/processed_annotations.json", "r")
)
outdir = Path(f"{BASEDIR}/1b/figures")
outdir.mkdir(exist_ok=True)
repaired_path = get_path(["paths", "jointemb_models"]) / ckpt_file_name.replace(
    ".ckpt", "_repaired.ckpt"
)
(
    pl_model_cellwhisperer,
    text_processor_cellwhisperer,
    transcriptome_processor_cellwhisperer,
) = load_cellwhisperer_model(model_path=repaired_path, eval=True)

# Prepare annData
adata = adata[
    ~np.isnan(npz["transcriptome_embeds"]).any(axis=1)
]  # 98 transcriptomes have nan values
embeds_not_nan = npz["transcriptome_embeds"][
    ~np.isnan(npz["transcriptome_embeds"]).any(axis=1)
]  # subset also the npz file for score calculation later
adata.obs["cluster_label"] = adata.obs["leiden"].apply(
    lambda x: f"{x} ({language_labels.iloc[int(x)]})"
)
adata.obs["cluster_label"] = [
    x[:60].replace("\n", "") for x in adata.obs["cluster_label"]
]
annotations_per_transcriptome_keys_list = list(annotations_per_transcriptome.keys())
adata.obs["natural_language_annotation"] = [
    annotations_per_transcriptome[annotations_per_transcriptome_keys_list[int(idx)]]
    for idx in adata.obs.index
]

# Prepare top terms per cluster
all_terms = df[df["library"] == library]["term_without_prefix"].unique().tolist()
scores, annot = score_transcriptomes_vs_texts(
    model=pl_model_cellwhisperer.model,
    transcriptome_input=torch.tensor(
        embeds_not_nan, device=pl_model_cellwhisperer.model.device
    ),
    text_list_or_text_embeds=all_terms,
    average_mode="embeddings",
    grouping_keys=adata.obs.cluster_label,
    transcriptome_processor=transcriptome_processor_cellwhisperer,
    batch_size=32,
    score_norm_method=score_norm_method,
)
df_scores = pd.DataFrame(index=annot, columns=all_terms, data=scores.T)
df_zscored_over_terms = (df_scores - df_scores.mean()) / df_scores.std()
df_scores["sum_of_all_scores"] = df_zscored_over_terms.sum(axis=1)
df_stack = df_scores.stack().reset_index()
df_stack.columns = ["cluster_label", "term_without_prefix", "logits"]
df_stack = df_stack.sort_values(by="logits", ascending=False)

# Rank the terms by the difference between the score of the top cluster and a reference cluster (e.g. the second best)
term_dict = {}
for term, df_this_term in df_stack.groupby("term_without_prefix"):
    df_this_term = df_this_term.sort_values(by="logits", ascending=False)
    score_diff = (
        df_this_term["logits"].iloc[0]
        - df_this_term["logits"].iloc[compare_top_vs_clusternumber]
    )
    term_dict[term] = {
        "term_without_prefix": term,
        "score_diff": score_diff,
        "top_cluster": df_this_term["cluster_label"].iloc[0],
        "top_score": df_this_term["logits"].iloc[0],
    }
df_score_diff = pd.DataFrame.from_dict(term_dict, orient="index").sort_values(
    by="score_diff", ascending=False
)

# keep at most n terms per cluster but keep order
df_score_diff = (
    df_score_diff.groupby("top_cluster")
    .head(max_terms_per_cluster)
    .sort_values(by="score_diff", ascending=False)
)
top_terms = df_score_diff["term_without_prefix"].tolist()[:max_terms_total]

top_terms += df_stack.drop_duplicates(subset="term_without_prefix")[
    "term_without_prefix"
][
    :20
].tolist()  # add the overall top 20 terms
top_terms = list(set(top_terms))  # remove duplicates

# Calculate the scores for all terms
scores_all_terms, _ = score_transcriptomes_vs_texts(
    model=pl_model_cellwhisperer.model,
    transcriptome_input=torch.tensor(
        embeds_not_nan, device=pl_model_cellwhisperer.model.device
    ),
    text_list_or_text_embeds=all_terms,
    average_mode=None,
    grouping_keys=None,
    transcriptome_processor=transcriptome_processor_cellwhisperer,
    batch_size=32,
    score_norm_method=score_norm_method,
)
scores_all_terms = scores_all_terms.T  # n_cells * n_text

# Calculate the sum of normalized scores over all terms
scores_zscored_over_terms = (
    scores_all_terms[:, 1:] - scores_all_terms[:, 1:].mean(axis=0)
) / scores_all_terms[:, 1:].std(axis=0)
normed_score_sum_of_all_terms = scores_zscored_over_terms.sum(axis=1)  # n_cells

scores_top_terms = scores_all_terms[:, [all_terms.index(term) for term in top_terms]]
top_terms = ["sum_of_all_scores"] + top_terms
scores_top_terms = torch.cat(
    [normed_score_sum_of_all_terms.unsqueeze(1), scores_top_terms], dim=1
)

## First, plot and save the umaps colored by clusters:
# Version 1: Full legend, wide figure
fig, ax = plt.subplots(figsize=(30, 10))
sc.pl.umap(
    adata,
    color=["cluster_label"],
    ax=ax,
    palette=list(matplotlib.colors.CSS4_COLORS.values()),
    show=False,
    legend_fontsize=6,
    s=0.75,
)
plt.subplots_adjust(right=0.5)
plt.savefig(outdir / "umap_clusters_all.pdf", dpi=660)
plt.close()
# Version 2: No legend, only cluster numbers
fig, ax = plt.subplots(figsize=(10, 10))
sc.pl.umap(
    adata,
    color=["leiden"],
    ax=ax,
    palette=list(matplotlib.colors.CSS4_COLORS.values()),
    legend_loc="on data",
    legend_fontsize=6,
    show=False,
    s=0.75,
)
plt.savefig(outdir / "umap_clusters_all.leidennumbers_only.pdf", dpi=660)
plt.close()

# Plot each of the term scores
for i, term in enumerate(top_terms):
    plot_term_score_umaps(
        adata=adata,
        term=term,
        scores_this_term=scores_top_terms[:, i].tolist(),
        df_stack=df_stack,
        outdir=outdir,
    )
