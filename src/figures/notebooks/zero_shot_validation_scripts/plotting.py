import scanpy as sc
import anndata
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os
import warnings
import matplotlib
from collections import defaultdict

sc.set_figure_params(vector_friendly=True, dpi_save=500)  # Makes PDFs of scatter plots much smaller in size but still high-quality

def plot_embeddings_with_scores(
    adata, analysis_types, result_metrics_dict, dataset_name, result_dir,
    celltype_plot_palette=None
):
    """
    Plot the embeddings colored by batch and celltype, and add the integration scores to the title.
    """
    fig, axes = plt.subplots(
        len(analysis_types), 2, figsize=(15, len(analysis_types) * 5)
    )
    if len(analysis_types) == 1:
        axes = [axes]
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
            palette=celltype_plot_palette,
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
        if adata.obs.celltype.nunique() > 50:
            axes[i][1].get_legend().remove()
    plt.tight_layout()
    plt.suptitle(dataset_name)
    for suffix in ["png", "pdf"]:
        plt.savefig(f"{result_dir}/embedding_plots_zero_shot_comparison.{suffix}", dpi=900 if suffix == "png" else None)
    plt.show()
    plt.close()


def plot_cellwhisperer_predictions_on_umap(
    adata: anndata.AnnData,
    result_dir: str,
    label_col="celltype",
    color_mapping=None,
    background_adata=None,
) -> None:
    """Plot the single-cellm predicted labels on the UMAP."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        embedding_basis = "X_umap_on_neighbors_cellwhisperer"

        if color_mapping is None:
            if f"{label_col}_colors" in adata.uns.keys():
                del adata.uns[f"{label_col}_colors"]
            # to assign colors to label_col_colors, plot once but don't show
            sc.pl.embedding(adata, basis=embedding_basis, color=[label_col, "batch"], frameon=False, s=10, alpha=0.5, legend_fontsize=8, legend_loc="right margin", legend_fontoutline=2,
                            show=False)
            plt.close()

            color_mapping = dict(zip(adata.obs[label_col].cat.categories, adata.uns[f"{label_col}_colors"]))
            color_mapping.update(dict(zip(adata.obs["batch"].cat.categories, adata.uns[f"batch_colors"])))

        adata.uns[f"predicted_labels_cellwhisperer_colors"] = adata.uns[
            f"{label_col}_colors"
        ]
        for color in [
                label_col,
                "predicted_labels_cellwhisperer",
                "batch",

        ]:
            ax=plt.gca()
            if background_adata is not None:
                # Plot the background in grey
                sc.pl.embedding(
                    background_adata,
                    basis=embedding_basis,
                    frameon=False,
                    s=10,
                    alpha=0.3,
                    legend_fontsize=6,
                    show=False,
                    # palette=color_mapping,
                    ncols=1,
                    ax=ax
                )

            sc.pl.embedding(
                adata[adata.obs[label_col].isin(list(color_mapping.keys()))],
                basis=embedding_basis,
                color=color,
                frameon=False,
                s=10,
                alpha=0.5,
                legend_fontsize=6,
                show=False,
                palette=color_mapping,
                ncols=1,
                ax=ax
            )
            plt.gcf().set_size_inches(10, 5)
            plt.subplots_adjust(right=0.55)
            for suffix in ["png", "pdf"]:
                plt.savefig(
                    f"{result_dir}/UMAP.{label_col}_as_label.{color}.{suffix}", dpi=900 if suffix == "png" else None
                )
            plt.show()
            plt.close()


def plot_confusion_matrix(
    performance_metrics_per_label_df: pd.DataFrame,
    result_dir: str,
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
        confusion_matrix.to_excel(f"{result_dir}/confusion_matrix_cellwhisperer.{label_col}_as_label.norm_{norm}.xlsx", index=True)

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
            ha="right",
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

        plt.savefig(
            f"{result_dir}/confusion_matrix_cellwhisperer.{label_col}_as_label.norm_{norm}.png", dpi=900
        )
        plt.savefig(
            f"{result_dir}/confusion_matrix_cellwhisperer.{label_col}_as_label.norm_{norm}.pdf")
        plt.show()
        plt.close()


def plot_term_search_result(term, celltype, adata, result_dir, prefix, suffix):
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
        plt.savefig(f"{result_dir}/umap_on_neighbors_cellwhisperer.true_celltype_{celltype}.{file_suffix}")
    plt.tight_layout()
    plt.show()
    plt.close()

    for make_colorscale_symmetrical in [True,False]:
        vmax=adata.obs[f"score_for_{term}"].max()
        sc.pl.embedding(adata, 
        basis="X_umap_on_neighbors_cellwhisperer" ,#if not "X_umap_original" in adata.obsm.keys() else "X_umap_original",
        color=[f"score_for_{term}"],
        cmap="RdBu_r", vmin=-vmax if make_colorscale_symmetrical else None, vmax=vmax if make_colorscale_symmetrical else None,show=False)
        plt.title("Keyword search results")
        # label the colorbar
        plt.gcf().axes[1].set_ylabel(f"Score for: '{prefix}{term}{suffix}'", fontsize=7)
        plt.gcf().axes[0].set_facecolor("white")


        for file_suffix in ["png","pdf"]:
            plt.savefig(f"{result_dir}/umap_on_neighbors_cellwhisperer.keyword_{term}.{'symmetrical_cmap' if make_colorscale_symmetrical else 'asymmetrical_cmap'}.{file_suffix}")
        plt.tight_layout()
        plt.show()
        plt.close()


def plot_confidence_distributions(adata, result_dir, dataset_name, text_list,
                                  label_col="celltype"):
    """Plot a number of histograms and KDEplots for the cellwhisperer score across different values for label_col"""
    

    hist_dfs_all_terms={"unnormed":[],"normed":[]}
    try: # can lead to errors if the number of unique labels is too high
        if len(adata.obs[label_col].unique()) < 1000:
            fig, ax = plt.subplots(len(adata.obs[label_col].unique()),1, sharex=True,sharey=False,figsize=(8,2*len(adata.obs[label_col].unique())))
            for i, term in enumerate(text_list):
                matching_label=adata.obs[label_col].unique().tolist()[i]
                adata.obs["label_matches_term"]=adata.obs[label_col]==matching_label
                sns.histplot(data=adata.obs,
                            x=f"score_for_{term}",
                            hue="label_matches_term",
                            ax=ax[i],bins=20,
                            stat="density",
                            common_norm=False,
                            palette={True:"coral",False:"silver"},
                            legend=False)
                hist_df=adata.obs[[f"score_for_{term}","label_matches_term"]]
                hist_df.columns=["score","label_matches_term"]
                hist_dfs_all_terms["unnormed"].append(hist_df.copy())


                plt.sca(ax[i])
                plt.legend(title=f"Cell type",labels=[matching_label,"other"],loc="lower right",
                        ncol=1)

                # z-normalize vs the label_matches_term = False
                hist_score_normed=hist_df.copy()
                mean=hist_score_normed[hist_score_normed["label_matches_term"]==False]["score"].mean()
                std=hist_score_normed[hist_score_normed["label_matches_term"]==False]["score"].std()
                hist_score_normed["score"]=(hist_score_normed["score"]-mean)/std
                hist_dfs_all_terms["normed"].append(hist_score_normed.copy())

            plt.xlabel("Cellwhisperer score for the label")
            plt.savefig(f"{result_dir}/confidence_distribution_{label_col}_per_label.pdf")
            plt.show()
            plt.close()
        
        for norm in ["unnormed","normed"]:
            hist_df_all_terms=pd.concat(hist_dfs_all_terms[norm])
            sns.histplot(data=hist_df_all_terms,
                                x=f"score",
                                hue="label_matches_term",
                                bins=20,
                                stat="density",
                                common_norm=False,
                                palette={True:"coral",False:"silver"},
                                legend=True)
            plt.xlabel(f"{'Normalized c' if norm=='normed' else 'C'}ellwhisperer score for the label")
            plt.ylabel("Density")
            plt.gca().get_legend().set_title("Cell type equals label")
            plt.savefig(f"{result_dir}/confidence_distribution_{label_col}_all_labels.{norm}.pdf")
            plt.show()
            plt.close()

        # Some specific examples
        if "tabula_sapiens" in dataset_name:
            fig, ax = plt.subplots(3,1, sharex=True,sharey=False,figsize=(8,2*3))
            for i, term in enumerate(["cardiac muscle cell","alveolar fibroblast","thymocyte", "erythrocyte"]):
                matching_label=adata.obs[label_col].unique().tolist()[i]
                adata.obs["label_matches_term"]=adata.obs[label_col]==matching_label
                sns.histplot(data=adata.obs,
                            x=f"score_for_{term}",
                            hue="label_matches_term",
                            ax=ax[i],bins=20,
                            stat="density",
                            common_norm=False,
                            palette={True:"coral",False:"silver"},
                            legend=False)
                hist_df=adata.obs[[f"score_for_{term}","label_matches_term"]]
                hist_df.columns=["score","label_matches_term"]
                hist_dfs_all_terms["unnormed"].append(hist_df.copy())
                plt.sca(ax[i])
            plt.legend(title=f"Cell type",labels=[matching_label,"other"],loc="lower right",
                        ncol=1)
            plt.xlabel("Cellwhisperer score for the label")
            plt.savefig(f"{result_dir}/confidence_distribution_{label_col}_per_label.SELECTED_TERMS.pdf")
            plt.show()
            plt.close()
            
    except Exception as e:
        print(f"Got the following error during plotting of confidence distributions (continueing): {e}")
        

    # Plot the distribution of confidence scores - seperately for cases where the prediction is correct vs incorrect
    sns.kdeplot(
        data=adata.obs,
        x="confidence_cellwhisperer",
        hue="correct_prediction",
        common_norm=False,
    )
    plt.savefig(f"{result_dir}//confidence_distribution_{label_col}.pdf")
    plt.close()

    sns.histplot(
        data=adata.obs,
        x="confidence_cellwhisperer",
        hue="correct_prediction",
        common_norm=False,
    )
    plt.savefig(f"{result_dir}//confidence_distribution_{label_col}_hist.pdf")
    plt.close()


def plot_integration_metrics(integration_scores_df, result_dir):
    """Bar plots of integration metrics for each method."""

    sns.barplot(
        data=integration_scores_df,
        x="metric",
        y="value",
        hue="Method",
        palette="Greys",
    )
    for i, row in integration_scores_df.iterrows():
        plt.text(x=i/2-0.25,y=row["value"]+0.01,s=f"{round(row['value'],2)}",ha="center",fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gcf().set_size_inches(5, 4)
    plt.ylim(0,1)
    plt.xlabel("")
    plt.ylabel("Score")

    plt.savefig(f"{result_dir}//integration_scores.pdf")
    plt.close()


def plot_performance_metrics_example_classes(result_dir, label_cols, datasets, selected_sample_lists, suffix_prefix_dict):
    """Barplots for AUROC and accuracy for the selected examples."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(2*len(datasets), 2), sharey=True)
    for i, label_col, dataset, selected_samples in zip(range(len(label_cols)), label_cols, datasets, selected_sample_lists):
        df_path = f"{result_dir}/{dataset}/performance_metrics_cellwhisperer.{label_col}_as_label.per_{label_col}.csv"
        prefix, suffix = suffix_prefix_dict[label_col]

        plt.sca(axes[i])     
        df = pd.read_csv(df_path)
        df["class"] = df["class"].str.replace(prefix,"").str.replace(suffix,"")
        plot_df = df[df["class"].isin(selected_samples)][["class","rocauc","accuracy"]].copy()
        plot_df = plot_df.rename(columns={"rocauc":"ROC-AUC","accuracy":"Accuracy"})
        plot_df = pd.melt(plot_df, id_vars="class", value_vars=["ROC-AUC","Accuracy"], var_name="metric", value_name="value")
        sns.barplot(data=plot_df, x="class", y="value", hue="metric", width=0.6)
        plt.axhline(y=1/len(df["class"].unique()), color=sns.color_palette()[1], linestyle="--", label="Accuracy (random baseline)")
        plt.axhline(y=0.5, color=sns.color_palette()[0], linestyle="--", label="AUROC (random baseline)")
        plt.title(dataset)
        plt.ylim(0,1)
        plt.legend()
        #plt.xlabel("")
        plt.ylabel("Score")

    plt.savefig(f"{result_dir}/performance_metrics_cellwhisperer.selected_classes_and_datasets.pdf")
    plt.tight_layout()
    plt.show()


def plot_performance_metrics_macro_avg(result_dir, label_cols, datasets):
    """Barplots for AUROC and accuracy for the selected datasets/label_cols."""
    scores = defaultdict(list)
    for i, label_col, dataset in zip(range(len(label_cols)),label_cols, datasets):
        df_path=f"{result_dir}/{dataset}/performance_metrics_cellwhisperer.{label_col}_as_label.macrovag.csv"
        df=pd.read_csv(df_path, index_col=0)
        per_class_path=f"{result_dir}/{dataset}/performance_metrics_cellwhisperer.{label_col}_as_label.per_{label_col}.csv"
        df_per_class=pd.read_csv(per_class_path)
        n_classes=len(df_per_class)

        scores["AUROC"].append(float(df.loc["rocauc_macroAvg"].item().replace("tensor(","").replace(")","")))
        scores["AUROC (random baseline)"].append(0.5)
        scores["Accuracy"].append(float(df.loc["accuracy_macroAvg"].item().replace("tensor(","").replace(")","")))
        scores["Accuracy (random baseline)"].append(1/n_classes)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    plt.sca(axes[0])
    plt.bar(range(len(scores["AUROC"])), scores["AUROC"],label="AUROC", color="#4d4d4dff")
    for i in range (len(scores["Accuracy (random baseline)"])):
        plt.plot([i-0.4,i+0.4],[scores["AUROC (random baseline)"][i]]*2, color="#1a1a1aff", linestyle="--",label="AUROC (random baseline)")
    for i, score in enumerate(scores["AUROC"]):
        plt.text(i,score+0.01,f"{round(score,2)}",ha="center", rotation=90)
    #plt.xticks([])
    
    plt.sca(axes[1])
    plt.bar(range(len(scores["Accuracy"])), scores["Accuracy"],label="Accuracy", color="#4d4d4dff")
    for i in range (len(scores["Accuracy (random baseline)"])):
        plt.plot([i-0.4,i+0.4],[scores["Accuracy (random baseline)"][i]]*2, color="#1a1a1aff", linestyle="--",label="Accuracy (random baseline)")
    for i, score in enumerate(scores["Accuracy"]):
        plt.text(i,score+0.01,f"{round(score,2)}",ha="center", rotation=90)
    #plt.xticks([])

    plt.savefig(f"{result_dir}/performance_metrics_cellwhisperer.selected_datasets.rocauc_and_accuracy.pdf")