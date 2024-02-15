from cellwhisperer.jointemb.cellwhisperer_lightning import (
    TranscriptomeTextDualEncoderLightning,
)
from cellwhisperer.validation.zero_shot.cancer_gene_essentiality import (
    evaluate_cancer_gene_essentiality,
)


def main():
    pl_model = TranscriptomeTextDualEncoderLightning()
    model = pl_model.model

    metrics, results_df = evaluate_cancer_gene_essentiality(model)
    model.store_cache()
    print(metrics, results_df)


if __name__ == "__main__":
    main()
