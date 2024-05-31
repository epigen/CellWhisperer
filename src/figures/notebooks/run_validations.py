import os
from pathlib import Path
import pandas as pd
import logging
import numpy as np
from collections import defaultdict
import copy
import matplotlib
import torch
import snakemake

from cellwhisperer.config import get_path
from cellwhisperer.utils.inference import score_transcriptomes_vs_texts
from cellwhisperer.validation.integration.functions import eval_scib_metrics
from cellwhisperer.validation.zero_shot.single_cell_annotation import (
    get_performance_metrics_transcriptome_vs_text
)
from cellwhisperer.utils.model_io import load_cellwhisperer_model
from server.common.colors import CSS4_NAMED_COLORS 
from scripts.utils import umap_on_embedding, prepare_integration_df,TABSAP_WELLSTUDIED_COLORMAPPING, PANCREAS_ORDER
from scripts.embedding_generation import get_adata_with_embedding
from scripts.dataset_preparation import load_and_preprocess_dataset
from scripts.plotting import (
    plot_embeddings_with_scores,
    plot_cellwhisperer_predictions_on_umap,
    plot_confusion_matrix,
    plot_term_search_result,
    plot_confidence_distributions,
    plot_integration_metrics,
    plot_performance_metrics_macro_avg,
    plot_performance_metrics_example_classes
)

matplotlib.style.use(get_path(["plot_style"]))

#%%
#### Parameters ####

ckpt_file_name=snakemake.params.model
ckpt_file_path = get_path(["paths", "jointemb_models"]) / ckpt_file_name

result_dir=snakemake.output.result_dir

dataset_names = snakemake.params.datasets


## Choose the datasets and analysis types to run
analysis_types = [
    "cellwhisperer",
    "geneformer",
    ]

## Select which columns to predict against. 
label_cols_per_dataset_dict = defaultdict(list)
label_cols_per_dataset_dict.update({x: ["celltype"] for x in dataset_names}) # always predict celltype

for dataset in dataset_names:
    if "tabula_sapiens" in dataset:
        label_cols_per_dataset_dict[dataset] += ["organ_tissue"]

label_cols_per_dataset_dict["human_disease"] += [
    "Tissue",
    "Tissue_subtype",
    "Disease",
    "Disease_subtype",
]

## Define the suffix and prefix for the text embeddings
suffix_prefix_dict = {}
suffix_prefix_dict["celltype"] = ("A sample of ", " from a healthy individual")
suffix_prefix_dict["organ_tissue"] = ("A sample of ", " from a healthy individual")
suffix_prefix_dict["Disease"] = ("A sample from an individual with ","")
suffix_prefix_dict["Disease_subtype"] = ("A sample from an individual with ","")
suffix_prefix_dict["Tissue"] = ("A "," sample")
suffix_prefix_dict["Tissue_subtype"] = ("A "," sample")

use_prefix_suffix_version = True

#%%
#### Load models
models_and_processors_dict = {}

# Load the proper cellwhisperer model
(
    pl_model_cellwhisperer,
    text_processor_cellwhisperer,
    transcriptome_processor_cellwhisperer,
) = load_cellwhisperer_model(model_path=ckpt_file_path, eval=True)
models_and_processors_dict["cellwhisperer"] = (
    pl_model_cellwhisperer.model,
    transcriptome_processor_cellwhisperer,
)

# Load a model with a pretrained Geneformer transcriptome model backbone
(
    model_w_geneformer,
    text_processor_geneformer,
    transcriptome_processor_geneformer,
) = load_cellwhisperer_model(
    model_path=None, eval=True, transcriptome_model_type="geneformer"
)
models_and_processors_dict["geneformer"] = (
    model_w_geneformer.model,
    transcriptome_processor_geneformer,
)
#%%
#### Iterate over datasets
for dataset_name in dataset_names:
    result_metrics_dict = {}
    logging.info(f"Starting with {dataset_name}")
    os.makedirs(f"{result_dir}/{dataset_name}", exist_ok=True)

    #### Load data
    adata = load_and_preprocess_dataset(dataset_name)
    print(adata.obs.celltype)
    logging.info(f"Data loaded and preprocessed. Shape: {adata.shape}")

    #### Create embeddings and calculate metrics
    for analysis_type in analysis_types:
        logging.info(f"Starting with {analysis_type}")

        # Add the embeddings to the adata:
        adata = get_adata_with_embedding(
            adata=adata,
            models_and_processors_dict=models_and_processors_dict,
            analysis_type=analysis_type,
        )

        # Calculate UMAPs based on the embeddings
        adata = umap_on_embedding(
            adata,
            embedding_key=f"X_{analysis_type}",
            neighbors_key=f"neighbors_{analysis_type}",
            umap_key=f"X_umap_on_neighbors_{analysis_type}",
        )

        # Calculate integration metrics
        result_metrics_dict[(dataset_name, analysis_type)] = eval_scib_metrics(
            adata,
            label_key="celltype",
            batch_key="batch",
            embedding_key=f"X_{analysis_type}",
        )

        logging.info(f"Finished with {analysis_type}")

    celltype_palette = {celltype:list(CSS4_NAMED_COLORS.values())[i if i<len(CSS4_NAMED_COLORS.values()) else i-len(CSS4_NAMED_COLORS.values())] for i,celltype in enumerate(adata.obs.celltype.unique())}
    if "tabula_sapiens" in dataset_name:
        # update the celltype palette with the well-studied cell types
        celltype_palette.update(TABSAP_WELLSTUDIED_COLORMAPPING)
    
    # Plot the embeddings generated by the different methods, colored by celltype and batch
    plot_embeddings_with_scores(
        adata=adata,
        analysis_types=analysis_types,
        result_metrics_dict=result_metrics_dict,
        dataset_name=dataset_name,
        result_dir=result_dir,
        celltype_plot_palette=celltype_palette,
    )

    if adata.obs.batch.nunique() > 1:
        integration_scores_df=prepare_integration_df(result_metrics_dict)
        #### Plot and Save integration metrics
        integration_scores_df.to_csv(f"{result_dir}/{dataset_name}/metrics_MS_zero_shot.csv")
        plot_integration_metrics(integration_scores_df, result_dir, dataset_name)

    if "tabula_sapiens" in dataset_name:
        color_mapping = copy.copy(TABSAP_WELLSTUDIED_COLORMAPPING)
    else:
        color_mapping = dict(zip(adata.obs["celltype"].cat.categories, adata.uns[f"celltype_colors"]))
    color_mapping.update(dict(zip(adata.obs["batch"].cat.categories, adata.uns[f"batch_colors"])))

    #### Predict the labels using cellwhisperer

    for label_col in label_cols_per_dataset_dict[dataset_name]:
        adata_no_nans = adata[
            ~(adata.obs[label_col].isna()) & ~(adata.obs[label_col] == "nan")
        ].copy()

        if use_prefix_suffix_version and label_col in suffix_prefix_dict:
            prefix, suffix = suffix_prefix_dict[label_col]
            text_list=[f"{prefix}{x}{suffix}" for x in adata_no_nans.obs[label_col].unique().tolist()]
        elif label_col not in suffix_prefix_dict:
            logging.warning(f"Label column {label_col} not found in suffix_prefix_dict, continuing without prefix/suffix")
            text_list = adata_no_nans.obs[label_col].unique().tolist()

        scores, _ = score_transcriptomes_vs_texts(
            model=models_and_processors_dict["cellwhisperer"][0],
            logit_scale=models_and_processors_dict["cellwhisperer"][0].discriminator.temperature.exp(),
            transcriptome_input=torch.tensor(adata_no_nans.obsm["X_cellwhisperer"],
                                                device=models_and_processors_dict["cellwhisperer"][0].device),
            text_list_or_text_embeds=text_list,
            average_mode=None,
            grouping_keys=None,
            transcriptome_processor=models_and_processors_dict["cellwhisperer"][1],
            text_tokenizer=text_processor_cellwhisperer,
            batch_size=32,
            score_norm_method=None,  
        )
        scores = scores.T  # n_cells * n_text
        predicted_labels = [
            adata_no_nans.obs[label_col].unique().tolist()[x]
            for x in scores.argmax(axis=1)
        ]
        for term in text_list:
            adata_no_nans.obs[f"score_for_{term}"] = scores[:, text_list.index(term)].tolist()
        adata_no_nans.obs["predicted_labels_cellwhisperer"] = predicted_labels
        adata_no_nans.obs["confidence_cellwhisperer"] = scores.max(axis=1).values
        adata_no_nans.obs["correct_prediction"] = adata_no_nans.obs["predicted_labels_cellwhisperer"] == adata_no_nans.obs[label_col]

        #### Plot the confidence distributions
        plot_confidence_distributions(adata_no_nans, result_dir, dataset_name, text_list,
                                label_col=label_col)

        #### Plot the cellwhisperer predicted labels
        # Put them in correct order
        if "well_studied_celltypes" in dataset_name and label_col=="celltype":
            adata_no_nans.obs["celltype"] = pd.Categorical(
                values=adata_no_nans.obs["celltype"],
                categories=list(TABSAP_WELLSTUDIED_COLORMAPPING.keys()),
                ordered=True)
            adata_no_nans.obs["predicted_labels_cellwhisperer"] = pd.Categorical(
                values=adata_no_nans.obs["predicted_labels_cellwhisperer"],
                categories=adata_no_nans.obs["celltype"].cat.categories)


        if "tabula_sapiens" in dataset_name and label_col == "celltype" and not "well_studied_celltypes" in dataset_name:
            # extra predictions for the TabSap dataset: Only predict for the well-studied cell types
            # This allows plotting them on the same UMAP as the full dataset
            adata_wellstudied = adata_no_nans[
                adata_no_nans.obs["celltype"].isin(TABSAP_WELLSTUDIED_COLORMAPPING.keys())
            ].copy()
            if use_prefix_suffix_version and label_col in suffix_prefix_dict:
                wellstudied_texts = [f"{prefix}{x}{suffix}" for x in TABSAP_WELLSTUDIED_COLORMAPPING.keys()]
            else:
                wellstudied_texts = list(TABSAP_WELLSTUDIED_COLORMAPPING.keys())

            textlist_idx_wellstudied=[text_list.index(x) for x in text_list if x in wellstudied_texts]
            textlist_wellstudied = [text_list[x] for x in textlist_idx_wellstudied]
            scores_wellstudied = scores[adata_no_nans.obs["celltype"].isin(TABSAP_WELLSTUDIED_COLORMAPPING.keys()), :]
            scores_wellstudied =   scores_wellstudied[:,textlist_idx_wellstudied]
            predicted_labels_wellstudied = [textlist_wellstudied[x].replace(suffix,"").replace(prefix,"") for x in scores_wellstudied.argmax(axis=1)]
            adata_wellstudied.obs["predicted_labels_cellwhisperer"] = predicted_labels_wellstudied
            plot_cellwhisperer_predictions_on_umap(
                adata=adata_wellstudied,
                result_dir=result_dir,
                dataset_name=dataset_name,
                label_col=label_col,
                color_mapping=color_mapping if label_col == "celltype" else None,
                background_adata=adata_no_nans[~adata_no_nans.obs["celltype"].isin(TABSAP_WELLSTUDIED_COLORMAPPING.keys())]
            ) 
        else:
            plot_cellwhisperer_predictions_on_umap(
            adata=adata_no_nans,
            result_dir=result_dir,
            dataset_name=dataset_name,
            label_col=label_col,
            color_mapping=color_mapping if label_col == "celltype" else None,
        )                       

        #### Get classification performance metrics for cellwhisperer
            
        correct_text_idx_per_transcriptome=[
                adata_no_nans.obs[label_col].unique().tolist().index(x)
                for x in adata_no_nans.obs[label_col].values
            ]
        shuffled_text_idx_per_transcriptome=copy.copy(correct_text_idx_per_transcriptome)
        np.random.shuffle(shuffled_text_idx_per_transcriptome)
            
        for shuffle in [False, True]:
            (
                performance_metrics,
                performance_metrics_per_label_df,
            ) = get_performance_metrics_transcriptome_vs_text(
                model=models_and_processors_dict["cellwhisperer"][0],
                transcriptome_input=torch.tensor(adata_no_nans.obsm["X_cellwhisperer"], device=models_and_processors_dict["cellwhisperer"][0].device),
                text_list_or_text_embeds=text_list,#adata_no_nans.obs[label_col].unique().tolist(),
                average_mode=None,
                grouping_keys=None,
                transcriptome_processor=models_and_processors_dict["cellwhisperer"][1],
                text_tokenizer=text_processor_cellwhisperer,
                batch_size=32,
                score_norm_method=None,
                correct_text_idx_per_transcriptome=correct_text_idx_per_transcriptome if not shuffle else shuffled_text_idx_per_transcriptome,
            )
            pd.Series(performance_metrics).to_csv(
                f"{result_dir}/{dataset_name}/performance_metrics_cellwhisperer.{label_col}_as_label.macrovag.{'RANDOM' if shuffle else ''}.csv"
            )
            performance_metrics_per_label_df.to_csv(
                f"{result_dir}/{dataset_name}/performance_metrics_cellwhisperer.{label_col}_as_label.per_{label_col}.{'RANDOM' if shuffle else ''}.csv"
            )

        ## Plot the confusion matrix
        if dataset_name =="pancreas":
            order=PANCREAS_ORDER
        elif "well_studied_celltypes" in dataset_name and label_col=="celltype":
            order = list(TABSAP_WELLSTUDIED_COLORMAPPING.keys())
        else :
            order=None

        performance_metrics_per_label_df_wo_prefix_suffix = performance_metrics_per_label_df.copy()
        performance_metrics_per_label_df_wo_prefix_suffix.index = [
            x.replace("Sample of a ", "").replace("A sample of ","").replace(" from a healthy individual","")
            for x in performance_metrics_per_label_df.index.values
        ]
        performance_metrics_per_label_df_wo_prefix_suffix.columns = [
            x.replace("Sample of a ", "").replace("A sample of ","").replace(" from a healthy individual","")
            for x in performance_metrics_per_label_df.columns.values
        ]
        try:
            title = f"$\\text{{ROC-AUC}}_{{macro}}={round(float(performance_metrics['rocauc_macroAvg']),2)}$"
            plot_confusion_matrix(
                performance_metrics_per_label_df=performance_metrics_per_label_df_wo_prefix_suffix,
                result_dir=result_dir,
                dataset_name=dataset_name,
                label_col=label_col,
                order=order,
                title=title
            )
        except ValueError as e:
            print(f"Got the following error during plotting of confusion matrix (continueing): {e}")

        del adata_no_nans


    ## Term search in tabula sapiens
    if "tabula_sapiens" in dataset_name and not "well_studied_celltypes" in dataset_name:

        prefix, suffix = suffix_prefix_dict["celltype"]

        terms_celltype_dict={
            "red blood cells": "erythrocyte",
            "erythrocyte": "erythrocyte",
            "Natural killer cells":"nk cell",
            "T cells":"t cell",
            "B cells":"b cell",
            "blood platelets":"platelet",
        }
        for term,celltype in terms_celltype_dict.items():

            scores, _ = score_transcriptomes_vs_texts(
            model=models_and_processors_dict["cellwhisperer"][0],
            logit_scale=models_and_processors_dict["cellwhisperer"][0].discriminator.temperature.exp(),
            transcriptome_input=torch.tensor(adata.obsm["X_cellwhisperer"], device=models_and_processors_dict["cellwhisperer"][0].device),
            text_list_or_text_embeds=[f"{prefix}{term}{suffix}"],
            average_mode=None,
            grouping_keys=None,
            transcriptome_processor=models_and_processors_dict["cellwhisperer"][1],
            text_tokenizer=text_processor_cellwhisperer,
            batch_size=32,
            score_norm_method=None,
            )
            scores = scores.T  # n_cells * n_text
            adata.obs[f"score_for_{term}"] = scores.squeeze().tolist()
            adata.obs[f"label contains '{celltype}'"] = adata.obs["celltype"].str.contains(celltype).astype(int)

            plot_term_search_result(term, celltype, adata, result_dir, dataset_name, prefix, suffix)

#%%
# Performance on some examples for various tasks:
plot_performance_metrics_example_classes(
    result_dir=result_dir,          
    label_cols=[
        "celltype",
        "organ_tissue",
        "Disease_subtype",
                ],
    datasets=[
        "tabula_sapiens",
        "tabula_sapiens",
        "human_disease",],
    selected_sample_lists=[
                    ["kidney epithelial cell", "erythrocyte","plasma cell"],
                    ["Kidney", "Lung", "Tongue"],
                    ["Dilated cardiomyopathy","Melanoma","Hepatocellular carcinoma"],
    ],
    suffix_prefix_dict=suffix_prefix_dict
)
#%%
plot_performance_metrics_macro_avg(
    result_dir=result_dir,
    label_cols=[
        "celltype",#"Tissue",
        "celltype",
        "celltype",
        "celltype",
        "Disease_subtype",
        "organ_tissue",
        "Tissue",],
    datasets=[
        "tabula_sapiens",
        "tabula_sapiens_well_studied_celltypes",
        "pancreas",
        "immgen",
        "human_disease",
        "tabula_sapiens",
        "human_disease"]
)