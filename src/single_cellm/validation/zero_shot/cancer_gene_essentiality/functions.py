from typing import Dict, Tuple
from single_cellm.jointemb.model import TranscriptomeTextDualEncoderModel
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
    cancer_sentence = "cancer"

    # load first transcriptome from our 15k dataset and perform in silico KOs
    datamodule = CancerGeneEssentialityDataModule(
        tokenizer=model.config.text_config.model_type,
        transcriptome_processor=model.config.transcriptome_config.model_type,
        dataset_name="daniel",
        batch_size=8,
    )

    # Use trainer for prediction only
    trainer = Trainer()
    predictions = trainer.predict(
        LitCancerGeneEssentiality(model, cancer_sentence), datamodule=datamodule
    )

    aggregated_predictions = {
        key: torch.cat([prediction[key].squeeze(0) for prediction in predictions])
        for key in predictions[0].keys()
    }
    df = pd.DataFrame(aggregated_predictions)
    if plot:
        sns.barplot(data=df, x="labels", y="similarity", ci="sd")

    return _log_reg_statistics(df)
