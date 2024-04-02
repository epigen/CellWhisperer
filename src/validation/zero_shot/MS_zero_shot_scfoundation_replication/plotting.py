import scanpy as sc
import anndata
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os
import warnings
import matplotlib

sc.set_figure_params(vector_friendly=True, dpi_save=300)  # Makes PDFs of scatter plots much smaller in size but still high-quality

def plot_embeddings_with_scores(
    adata, analysis_types, result_metrics_dict, dataset_name, result_dir
):
    """
    Plot the embeddings colored by batch and celltype, and add the scib scores to the title.
    """
    fig, axes = plt.subplots(
        len(analysis_types), 2, figsize=(15, len(analysis_types) * 5)
    )
    for i, analysis_type in enumerate(analysis_types):
        sc.pl.embedding(
            adata,
            basis=f"X_umap_on_neighbors_{analysis_type}",
            color="batch",
            frameon=False,
            s=10,
            alpha=0.5,
            legend_fontsize=8,
            legend_loc="right margin",
            legend_fontoutline=2,
            ax=axes[i][0],
            show=False,
        )
        try:
            batch_integration_score = round(
                result_metrics_dict[(dataset_name, analysis_type)]["ASW_label__batch"], 2
            )
        except KeyError:
            batch_integration_score = "NA"
        axes[i][0].set_title(
            f"{analysis_type}: batch \n batch integration score= {batch_integration_score}"
        )

        sc.pl.embedding(
            adata,
            basis=f"X_umap_on_neighbors_{analysis_type}",
            color="celltype",
            frameon=False,
            s=10,
            alpha=0.5,
            legend_fontsize=8,
            legend_loc="right margin",
            legend_fontoutline=2,
            ax=axes[i][1],
            show=False,
        )
        asw_label = round(
            result_metrics_dict[(dataset_name, analysis_type)]["ASW_label"], 2
        )
        avg_bio = round(
            result_metrics_dict[(dataset_name, analysis_type)]["avg_bio"], 2
        )
        axes[i][1].set_title(
            f"{analysis_type}: celltype\n ASW_label= {asw_label}\n avg_bio= {avg_bio}"
        )
    plt.tight_layout()
    plt.suptitle(dataset_name)
    os.makedirs(os.path.dirname(f"{result_dir}/{dataset_name}/"), exist_ok=True)
    for suffix in ["png", "pdf"]:
        plt.savefig(f"{result_dir}/{dataset_name}/embedding_plots_MS_zero_shot.{suffix}", dpi=900 if suffix == "png" else None)
    plt.show()
    plt.close()


def plot_cellwhisperer_predictions_on_umap(
    adata: anndata.AnnData,
    result_dir: str,
    dataset_name: str,
    label_col="celltype",
    color_mapping=None,
) -> None:
    """Plot the single-cellm predicted labels in 2 versions: On the HVG-based umap and on the single-ceLLM based UMAP."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        embedding_basis = [
            "X_umap_on_neighbors_cellwhisperer",
        #    "X_umap_on_neighbors_hvg_without_PCA",
        ]


        if color_mapping is None:
            if f"{label_col}_colors" in adata.uns.keys():
                del adata.uns[f"{label_col}_colors"]
            # to assign colors to label_col_colors, plot once but don't show
            sc.pl.embedding(adata, basis=embedding_basis[0], color=[label_col, "batch"], frameon=False, s=10, alpha=0.5, legend_fontsize=8, legend_loc="right margin", legend_fontoutline=2,
                            show=False)
            plt.close()

            color_mapping = dict(zip(adata.obs[label_col].cat.categories, adata.uns[f"{label_col}_colors"]))
            color_mapping.update(dict(zip(adata.obs["batch"].cat.categories, adata.uns[f"batch_colors"])))

        adata.uns[f"predicted_labels_cellwhisperer_colors"] = adata.uns[
            f"{label_col}_colors"
        ]
        for basis in embedding_basis:
            for color in [
                    label_col,
                    "predicted_labels_cellwhisperer",
                    "batch",

            ]:
                sc.pl.embedding(
                    adata,
                    basis=basis,
                    color=color,
                    frameon=False,
                    s=10,
                    alpha=0.5,
                    legend_fontsize=6,
                    show=False,
                    palette=color_mapping,
                    ncols=1
                )
                os.makedirs(os.path.dirname(f"{result_dir}/{dataset_name}/"), exist_ok=True)
                plt.gcf().set_size_inches(10, 5)
                plt.subplots_adjust(right=0.55)
                for suffix in ["png", "pdf"]:
                    plt.savefig(
                        f"{result_dir}/{dataset_name}/cellwhisperer_predictions.{label_col}_as_label.{basis}.{color}.{suffix}", dpi=900 if suffix == "png" else None
                    )
                plt.show()
                plt.close()


def plot_confusion_matrix(
    performance_metrics_per_label_df: pd.DataFrame,
    result_dir: str,
    dataset_name: str,
    label_col="celltype",
    order=None,
    title=None,
) -> None:
    """Plot a heatmap of the confusion matrix in 2 versions: normalized and not normalized."""
    for norm in [True, False]:
        confusion_matrix = performance_metrics_per_label_df[
            [x for x in performance_metrics_per_label_df if x.startswith("n_samples_predicted_as_")]
        ]
        if norm:
            confusion_matrix = confusion_matrix.div(
                confusion_matrix.sum(axis=1), axis=0
            )
        confusion_matrix.columns = [
                x.replace("n_samples_predicted_as_", "")
                for x in confusion_matrix.columns
            ]
        if order is not None:
            confusion_matrix = confusion_matrix[order]
            confusion_matrix = confusion_matrix.loc[order]

        plt.figure(figsize=(10, 10))
        sns.heatmap(confusion_matrix, cmap="Blues", annot=False, square=True,
                    cbar_kws={"shrink": .7})
        plt.yticks(
            [x + 0.5 for x in range(len(confusion_matrix.index))],
            confusion_matrix.index,
        )
        plt.xticks(
            [x + 0.5 for x in range(len(confusion_matrix.columns))],
            confusion_matrix.columns,
            rotation=45,
        )
        plt.xlabel("Best-matching keyword")
        plt.ylabel("True class")
        plt.tight_layout()
        cbar = plt.gca().collections[0].colorbar
        cbar.set_label("Fraction of cells in true class")

        # mark the diagonal with boxes around the cells:
        for i in range(len(confusion_matrix.index)):
            for j in range(len(confusion_matrix.columns)):
                if i == j:
                    plt.gca().add_patch(
                        plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="grey", lw=2)
                    )

        plt.gcf().set_size_inches(
            max(5,len(confusion_matrix.index) // 2), max(5,len(confusion_matrix.index) // 2)
        )
        plt.title(title)

        os.makedirs(os.path.dirname(f"{result_dir}/{dataset_name}/"), exist_ok=True)
        plt.savefig(
            f"{result_dir}/{dataset_name}/confusion_matrix_cellwhisperer.{label_col}_as_label.norm_{norm}.png", dpi=900
        )
        plt.savefig(
            f"{result_dir}/{dataset_name}/confusion_matrix_cellwhisperer.{label_col}_as_label.norm_{norm}.pdf")
        plt.show()
        plt.close()


def plot_keyword_occurance_vs_performance(
    performance_metrics_per_label_df: pd.DataFrame,
    keyword_occurance_dict: dict,
    result_dir: str,
    dataset_name: str,
    label_col="celltype",
) -> None:
    """Plot a scatterplot of the number of times a keyword appears in the dataset vs. the performance of the model on that keyword."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        for score in ["f1", "rocauc", "recall_at_1", "recall_at_10"]:
            if performance_metrics_per_label_df[score].isna().all():
                continue
            for logx in [True, False]:
                plt.figure(figsize=(10, 10))
                sns.scatterplot(
                    x=list(keyword_occurance_dict.values()),
                    y=performance_metrics_per_label_df[score],
                    size=5,
                )
                # dont show legend:
                plt.gca().get_legend().remove()
                # label the points:
                for i, txt in enumerate(keyword_occurance_dict.keys()):
                    plt.annotate(
                        txt,
                        (
                            list(keyword_occurance_dict.values())[i],
                            performance_metrics_per_label_df[score].values[i],
                        ),
                        fontsize=6,
                    )
                # add a regression line and show pearson r and kendall tau:
                sns.regplot(
                    x=list(keyword_occurance_dict.values()),
                    y=performance_metrics_per_label_df[score],
                    scatter=False,
                )
                pearson_r = performance_metrics_per_label_df[score].corr(
                    pd.Series(keyword_occurance_dict)
                )
                kendall_tau = performance_metrics_per_label_df[score].corr(
                    pd.Series(keyword_occurance_dict), method="kendall"
                )
                plt.title(
                    f"{score} vs. keyword occurance in the dataset\n Pearson r= {pearson_r}\n Kendall tau= {kendall_tau}"
                )
                plt.xlabel("Number of times the keyword appears in the dataset")
                plt.ylabel(score)
                plt.title(f"{score} vs. keyword occurance in the dataset")
                if logx:
                    plt.xscale("log")
                plt.gcf().set_size_inches(
                    max(5,len(keyword_occurance_dict.keys()) // 3),
                    max(5,len(keyword_occurance_dict.keys()) // 3),
                )
                plt.ylim(-0.05, 1.05)
                os.makedirs(
                    os.path.dirname(f"{result_dir}/{dataset_name}/"), exist_ok=True
                )
                plt.savefig(
                    f"{result_dir}/{dataset_name}/{score}_vs_keyword_occurance.{label_col}_as_label.logx_{logx}.png"
                )
                plt.show()
                plt.close()

def plot_best_worst_and_all_celltype_performances(performance_metrics_per_label_df, performance_metrics, metric, result_dir, dataset_name, label_col):
    """
    Plot a barplot of the 20 best and 20 worst-performing celltypes, and a barplot of all celltypes.
    """
    for show_all in [True, False]:
        n_types=performance_metrics_per_label_df.shape[0]
        if n_types>40 and show_all==False:
            plot_df=performance_metrics_per_label_df.sort_values(by=metric,ascending=False).iloc[list(range(0,20))+list(range(n_types-20,n_types))]
        else:
            plot_df=performance_metrics_per_label_df.sort_values(by=metric,ascending=False)
        plot_df=plot_df[[metric]]
        sns.barplot(data=plot_df.reset_index(),y=metric,
                    x="class", color="darkgrey")
        plt.axhline(float(performance_metrics[f"{metric}_macroAvg"]),
                                                            color="black", linestyle="--")
        if not show_all:
            plt.xticks(rotation=90)
        else:
            plt.xticks([])
        plt.tight_layout()
        if show_all and n_types>40:
            plt.gcf().set_size_inches(20,20)
        plt.savefig(f"{result_dir}/{dataset_name}/performance_metrics_cellwhisperer.{label_col}_as_label.best_and_worst_performing.showall{show_all}.metric_{metric}.png")
        plt.close()

def plot_term_search_result(term, celltype, adata, result_dir, dataset_name, prefix, suffix):
    """
    Plot the ground truth celltype and the keyword search results on the UMAP.
    """
    sc.pl.embedding(adata, 
                    basis="X_umap_on_neighbors_cellwhisperer",
                    color=[f"label contains '{celltype}'"],
                    cmap = matplotlib.colors.ListedColormap(['silver', 'firebrick']),
                    show=False)
    plt.title("Ground truth label")
    plt.gcf().axes[0].set_facecolor("white")
    plt.gcf().axes[1].remove()

    for file_suffix in ["png","pdf"]:
        plt.savefig(f"{result_dir}/{dataset_name}/umap_on_neighbors_cellwhisperer.true_celltype_{celltype}.{file_suffix}")
    plt.tight_layout()
    plt.show()
    plt.close()

    vmax=adata.obs[f"score_for_{term}"].max()
    sc.pl.embedding(adata, 
    basis="X_umap_on_neighbors_cellwhisperer" ,#if not "X_umap_original" in adata.obsm.keys() else "X_umap_original",
    color=[f"score_for_{term}"],
    cmap="RdBu_r", vmin=-vmax, vmax=vmax,show=False)
    plt.title("Keyword search results")
    # label the colorbar
    plt.gcf().axes[1].set_ylabel(f"Score for: '{prefix}{term}{suffix}'", fontsize=7)
    plt.gcf().axes[0].set_facecolor("white")


    for file_suffix in ["png","pdf"]:
        plt.savefig(f"{result_dir}/{dataset_name}/umap_on_neighbors_cellwhisperer.keyword_{term}.{file_suffix}")
    plt.tight_layout()
    plt.show()
    plt.close()