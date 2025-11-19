


## Quality Statistics
- Overall data quality: 71.03% high-quality cells (96,142 out of 135,346 mapped cells)
- High-quality cores (>95% good quality): 35 cores out of 69 total
- Samples with exactly 2 HQ cores: 16 samples
- Available candidate samples: 12 samples (where both cores are not currently used for validation)

## Training-Validation core split

We split to have intentional sample-level data leakage to make the task easier.


| Sample ID | Validation Core | Training Core | Validation Core Stats | Training Core Stats |
|-----------|-----------------|---------------|----------------------|---------------------|
| 010 | A10 | A9 | 100.0% HQ, 903 cells | 100.0% HQ, 1,557 cells |
| 277 | E8 | E7 | 100.0% HQ, 3,882 cells | 97.7% HQ, 3,074 cells |
| 303 | J4 | J3 | 100.0% HQ, 2,066 cells | 100.0% HQ, 1,454 cells |
| 385 | J5 | J6 | 100.0% HQ, 4,420 cells | 100.0% HQ, 5,280 cells |
| 432 | K2 | K1 | 100.0% HQ, 788 cells | 100.0% HQ, 85 cells |

Key Features of This Selection

✅ All validation cores are high-quality (>95% good quality cells)
✅ All training cores are high-quality (>95% good quality cells)
✅ Each validation core has a paired training core from the same sample_id
✅ No overlap with current validation cores
✅ Good cell count distribution (ranging from 788 to 4,420 cells per validation core)

### old split (not controlled for sample they belong to)

`validation_cores=["B9", "C1", "C2", "G2", "H2"]  # High-quality cores (>95% good_quality)`
