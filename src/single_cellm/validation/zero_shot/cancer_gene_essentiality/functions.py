from typing import Dict, Tuple
from single_cellm.jointemb.model import TranscriptomeTextDualEncoderModel
from single_cellm.config import get_path, model_path_from_name
from transformers import AutoTokenizer
from lightning import Trainer
import pandas as pd
import seaborn as sns
import numpy as np
import logging


from .dataset import (
    CancerGeneEssentialityDataModule,
)

from .model import LitCancerGeneEssentiality
import torch
import statsmodels.api as sm


def _log_reg_statistics(df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Assess how well our shared embedding similarities are able to predict cancer gene essentiality (using a logistic regression model)


    :param df: a dataframe with columns "labels" and "similarity"
    :return: a dictionary with metrics describing the predictability of the labels
    """

    # Add an intercept column to the dataframe
    df = df.copy()
    df["intercept"] = 1

    # Define the predictor variable(s) and the outcome variable
    X = df[["intercept", "similarity"]]
    y = df["labels"]

    # Fit the logistic regression model
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()

    # Get predictions
    df["predictions"] = result.predict(X)

    # Calculate accuracy metrics
    df["prediction_correct"] = (df["predictions"].round() == df["labels"]).astype(int)
    accuracy = df["prediction_correct"].mean()

    # Return odds ratio
    metrics = {
        "intercept_coeff": result.params.intercept,
        "var_coeff": result.params.similarity,  # how well does the embedding similarity predict the labels?
        "var_p_values": result.pvalues.similarity,  # how well does the embedding similarity predict the labels?
        "pseudo_r_squared": result.prsquared,  # goodness-of-fit of the model
        "log_likelihood": result.llf,  # goodness-of-fit of the model
        "AIC": result.aic,  # goodness-of-fit of the model
        "BIC": result.bic,  # goodness-of-fit of the model
        "accuracy": accuracy,  # how well does the embedding similarity predict the labels?
    }

    return metrics, df


def _embed_cancer_sentence(
    model: TranscriptomeTextDualEncoderModel, anchor_sentence="cancer"
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path_from_name(model.config.text_config.model_type)
    )
    # alterantive: tokenizer = self.trainer.datamodule.processor.tokenizer

    token_dict = tokenizer(anchor_sentence, return_tensors="pt", padding=True)
    token_dict = {k: v.to(model.device) for k, v in token_dict.items()}
    with torch.no_grad():
        anchor_embed = model.get_text_features(**token_dict, normalize_embeds=True)[1]
    return anchor_embed


def evaluate_cancer_gene_essentiality(
    model: TranscriptomeTextDualEncoderModel, plot=False
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Assess zero-shot cancer gene essentiality prediction performance:

    Knockout genes (set expression to 0), embed them, and compute the similarity to the cancer sentence.
    Expectation: essential genes should have a lower similarity to the cancer sentence than non-essential genes.

    :param model: a trained model

    """
    logging.info("Performing gene cancer essentiality evaluation...")

    # load first transcriptome from our 15k dataset and perform in silico KOs
    datamodule = CancerGeneEssentialityDataModule(
        tokenizer=model.config.text_config.model_type,
        transcriptome_processor=model.config.transcriptome_config.model_type,
        dataset_name="daniel",
        batch_size=8,
    )
    # Follow lightning API
    datamodule.prepare_data()
    datamodule.setup()
    dataloader = datamodule.predict_dataloader()

    model = model.eval()

    anchor_embed = _embed_cancer_sentence(model)

    # Iterate over dataloader, predict and calculate similarities
    similarities = []
    labels = []
    for batch in dataloader:
        labels.append(batch.pop("labels"))
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            transcriptome_embeds = model.get_transcriptome_features(
                **batch, normalize_embeds=True
            )[1]
        similarity = torch.matmul(
            anchor_embed, transcriptome_embeds.t().to(anchor_embed.device)
        )
        similarities.append(similarity.cpu())

    df = pd.DataFrame(
        {
            "similarity": torch.cat([sim.squeeze(0) for sim in similarities]).numpy(),
            "labels": torch.cat([lab.squeeze(0) for lab in labels]).numpy(),
        }
    )
    if plot:
        sns.barplot(data=df, x="labels", y="similarity", ci="sd")

    return _log_reg_statistics(df)
