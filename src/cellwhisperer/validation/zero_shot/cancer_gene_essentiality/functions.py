from typing import Dict, Tuple
from cellwhisperer.jointemb.model import TranscriptomeTextDualEncoderModel
from cellwhisperer.config import get_path, model_path_from_name
from transformers import AutoTokenizer
from lightning import Trainer
import pandas as pd
import seaborn as sns
import numpy as np
import logging
from torchmetrics.functional import f1_score
from scipy.stats import mannwhitneyu
from torchmetrics.classification import BinaryF1Score

from .dataset import (
    CancerGeneEssentialityDataModule,
)

import torch
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def _per_cell_line_mannwhineyu(
    df: pd.DataFrame,
    stat_dim: str = "gene_ko",
) -> float:
    """
    stat_dim: The dimension to perform the statistical test over. One of {gene_ko, cell_line}.

    # TODO we might want to z-score normalize before mean-aggregating
    """
    assert stat_dim in ["gene_ko", "cell_line"]

    # For each cell line, take the average similarity to "cancer" for the essential genes and the non-essential genes. Calculate the delta of those 2 values (mean(essential) - mean(non-essential))
    cellline_differences = df.groupby(["essential", stat_dim])["similarity"].mean()
    # Test whether the distribution of deltas across cell lines is significantly different beteen essential and non-essential genes
    stat, p_value = mannwhitneyu(
        cellline_differences.loc[True],
        cellline_differences.loc[False],
        alternative="less",
    )

    return -np.log10(p_value)


def _per_cell_line_f1_score_mean(df: pd.DataFrame) -> float:
    """
    filter the reference (WT, no KO) and compare it with the essential and non-essential ones, (closer, less close), leading to separate F1-scores
    """
    # Calculate F1 scores based on reference
    f1_scores = []

    # Initialize the F1Score metric
    f1_metric = BinaryF1Score()

    # references are calculated as negative controls (reference is where gene_ko == -1)
    references = df.loc[df.gene_ko == -1].set_index("cell_line")["similarity"]

    # references = df.groupby(["cell_line"])["similarity"].mean()

    for cell_line, reference in references.items():
        # Get the subset of the dataframe for the current cell line
        df_subset = df[(df["cell_line"] == cell_line) & (df["gene_ko"] != -1)]

        # Predictions are based on whether the similarity score is below the reference
        predictions = torch.tensor(
            df_subset["similarity"].values < reference, dtype=torch.int
        )

        # Targets are based on whether the gene is essential
        targets = torch.tensor(df_subset["essential"].values, dtype=torch.int)

        # Compute the F1 score using torchmetrics
        f1 = f1_metric(predictions, targets).item()

        # Store the F1 score for the current cell line
        f1_scores.append(f1)

        # Reset the metric for the next calculation (not sure if necessary)
        f1_metric.reset()

    return np.mean(f1_scores)


def _log_reg_statistics(df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Assess how well our shared embedding similarities are able to predict cancer gene essentiality (using a logistic regression model)


    :param df: a dataframe with columns "essential" and "similarity"
    :return: a dictionary with metrics describing the predictability of the essential
    """

    # Add an intercept column to the dataframe
    df = df.copy()
    df["intercept"] = 1

    # Define the predictor variable(s) and the outcome variable
    X = df[["intercept", "similarity"]]
    y = df["essential"]

    # Fit the logistic regression model
    logit_model = sm.Logit(y, X)
    try:
        result = logit_model.fit(disp=0)
    except np.linalg.LinAlgError as e:
        logger.warning(
            f"LinAlgError: {e}. Returning empty metrics for Cancer Gene Essentiality."
        )
        df["predictions"] = np.nan
        df["prediction_correct"] = False
        return {}, df

    # Get predictions
    df["predictions"] = result.predict(X)

    # Calculate accuracy metrics
    df["prediction_correct"] = (df["predictions"].round() == df["essential"]).astype(
        int
    )
    accuracy = df["prediction_correct"].mean()
    try:
        f1 = f1_score(
            torch.tensor(df["essential"].values),
            torch.tensor(df["predictions"].round().values),
            task="binary",
        )
    except Exception as e:
        logger.warning(
            f"Error calculating f1 score: {e}. Returning empty metrics for Cancer Gene Essentiality."
        )
        f1 = np.nan

    # Return odds ratio
    metrics = {
        "intercept_coeff": result.params.intercept,
        "var_coeff": result.params.similarity,  # how well does the embedding similarity predict the essential?
        "var_p_values": result.pvalues.similarity,  # how well does the embedding similarity predict the essential?
        "pseudo_r_squared": result.prsquared,  # goodness-of-fit of the model
        "log_likelihood": result.llf,  # goodness-of-fit of the model
        "AIC": result.aic,  # goodness-of-fit of the model
        "BIC": result.bic,  # goodness-of-fit of the model
        "accuracy": accuracy,  # how well does the embedding similarity predict the essential?
        "f1": f1,
    }

    return metrics, df


class EvaluateCancerGeneEssentiality:
    def __init__(self, batch_size, transcriptome_model_type, text_model_type):
        # load first transcriptome from our 15k dataset and perform in silico KOs
        self.anchor_sentence = "cancer cell line"  # or "cancer"
        datamodule = CancerGeneEssentialityDataModule(
            tokenizer=text_model_type,
            transcriptome_processor=transcriptome_model_type,
            dataset_name="ccle",
            batch_size=batch_size,
            transcriptome_processor_kwargs={},
        )

        # Follow lightning API
        datamodule.prepare_data()
        datamodule.setup()
        self.dataloader = datamodule.predict_dataloader()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path_from_name(text_model_type)
        )

    def _embed_cancer_sentence(
        self, model: TranscriptomeTextDualEncoderModel
    ) -> torch.Tensor:
        token_dict = self.tokenizer(
            self.anchor_sentence, return_tensors="pt", padding=True
        )
        token_dict = {k: v.to(model.device) for k, v in token_dict.items()}
        with torch.no_grad():
            anchor_embed = model.get_text_features(**token_dict, normalize_embeds=True)[
                1
            ]
        return anchor_embed

    def __call__(
        self, model: TranscriptomeTextDualEncoderModel, plot: bool = False
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Assess zero-shot cancer gene essentiality prediction performance:

        Knockout genes (set expression to 0), embed them, and compute the similarity to the cancer sentence.
        Expectation: essential genes should have a lower similarity to the cancer sentence than non-essential genes.

        :param model: a trained model

        """
        logger.info("Performing gene cancer essentiality evaluation...")

        model = model.eval()
        anchor_embed = self._embed_cancer_sentence(model)

        # Iterate over dataloader, predict and calculate similarities
        similarities = []
        essential = []
        gene_ko = []
        cell_line = []
        for batch in self.dataloader:
            essential.append(batch.pop("essential"))
            gene_ko.append(batch.pop("gene_ko"))
            cell_line.append(batch.pop("cell_line"))
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                _, transcriptome_embeds = model.get_transcriptome_features(
                    **batch, normalize_embeds=True
                )
            similarity = torch.matmul(
                anchor_embed, transcriptome_embeds.t().to(anchor_embed.device)
            )
            similarities.append(similarity.cpu())

        df = pd.DataFrame(
            {
                "similarity": torch.cat(
                    [sim.squeeze(0) for sim in similarities]
                ).numpy(),
                "essential": torch.cat([es.squeeze(0) for es in essential]).numpy(),
                "gene_ko": torch.cat([ko.squeeze(0) for ko in gene_ko]).numpy(),
                "cell_line": torch.cat([cl.squeeze(0) for cl in cell_line]).numpy(),
            }
        )
        if plot:
            sns.barplot(data=df, x="essential", y="similarity", ci="sd")

        # return _log_reg_statistics(df)

        mannwhitneyu_gene_neglogp = _per_cell_line_mannwhineyu(df, "gene_ko")
        mannwhitneyu_cellline_neglogp = _per_cell_line_mannwhineyu(df, "cell_line")

        per_cell_line_f1_mean = _per_cell_line_f1_score_mean(df)

        return {
            "mannwhitneyu_gene_neglogp": mannwhitneyu_gene_neglogp,
            "mannwhitneyu_cellline_neglogp": mannwhitneyu_cellline_neglogp,
            "per_cell_line_f1_mean": per_cell_line_f1_mean,
        }, df
