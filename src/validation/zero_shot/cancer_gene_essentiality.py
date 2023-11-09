from single_cellm.jointemb.single_cellm_lightning import (
    TranscriptomeTextDualEncoderLightning,
)
from single_cellm.validation.zero_shot.cancer_gene_essentiality import (
    evaluate_cancer_gene_essentiality,
)


def main():
    pl_model = TranscriptomeTextDualEncoderLightning()
    model = pl_model.model

    odds_ratio_metric = evaluate_cancer_gene_essentiality(model)
    print(odds_ratio_metric)


if __name__ == "__main__":
    main()
