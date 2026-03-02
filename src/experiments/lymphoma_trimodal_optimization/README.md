# About

This is the second round of model optimization, following the one done in `/home/moritz/code/cellwhisperer/src/experiments/918-cosmx1k-eval` and the one in `/home/moritz/code/cellwhisperer/src/experiments/933-**`

We test additional tweaks one by one through delta configs layered on `base_config.yaml` (wandb group: `lymphoma-trimodal-opt`).

Usage: `cellwhisperer fit --config base_config.yaml --config delta_config/<name>.yaml`

## Delta Configs

### Model

| Config | What it tests | Key changes |
|--------|--------------|-------------|
| `census_downweight.yaml` | Reduce cellxgene_census weight (~0.33x) | Duplicates lymphoma train datasets in `dataset_names` so census contributes ~1/9 of training |
| `larger_cnn.yaml` | Larger cell-level CNN | `cnn_embedding_dim`: 512→1024, `cnn_num_layers`: 4→6 |

### Data — Low-quality core filtering

Uses `valid_core` column (int 0/1, per-core annotation) to drop invalid cores. Filtered datasets (`*_hqcores`) created via `filter_good_quality` rule in `lymphoma_cosmx_large/Snakefile`.

| Dataset | Original | Filtered | Retained |
|---------|----------|----------|----------|
| TMA2_hqcores | 70,845 | 57,609 | 81.3% |
| TMA4_hqcores | 95,065 | 64,039 | 67.4% |
| TMA5_hqcores | 90,602 | 52,670 | 58.1% |

| Config | Train | Eval | Purpose |
|--------|-------|------|---------|
| `hqcores_train.yaml` | TMA4,5 (filtered) + census | TMA2 (filtered) | Effect of clean training data |
| `hqcores_ref.yaml` | TMA4,5 (unfiltered) + census | TMA2 (filtered) | Baseline (same eval set, dirty training) |

### Data — FNA filtering

Drops fine needle aspirate cores using `TISS_prefix == 'SHF'`. Filtered datasets (`*_nofna`) created via `filter_fna` rule in `lymphoma_cosmx_large/Snakefile`.

| Dataset | Original | Filtered | Retained | FNA cores dropped |
|---------|----------|----------|----------|-------------------|
| TMA2_nofna | 70,845 | 67,213 | 94.9% | 11 |
| TMA4_nofna | 90,602 | 64,721 | 71.4% | 29 |
| TMA5_nofna | 95,065 | 78,899 | 83.0% | 21 |

| Config | Train | Eval | Purpose |
|--------|-------|------|---------|
| `nofna_train.yaml` | TMA4,5 (no FNA) + census | TMA2 (no FNA) | Effect of removing FNA from training |
| `nofna_ref.yaml` | TMA4,5 (unfiltered) + census | TMA2 (no FNA) | Baseline (same eval set, FNA in training) |

### Data — Raw read counts

Existing h5ads have **SCTransform corrected counts** in `adata.X` (integer but not real raw counts — totals match `nCount_SCT`, not `nCount_RNA`). The `data` layer has log-normalized values, `scale` has SCT scale factors. The `ensure_raw_counts_adata()` function in the MLP processor passes these through since they look like integers, but they are not the instrument read counts.

Raw read counts are available in CosMx flat files at `/oak/.../CosMX/stanford_CART_TMA/data/*/flatFiles/*/exprMat_file.csv.gz`. The `replace_with_raw_counts` rule in `lymphoma_cosmx_large/Snakefile` replaces X with these raw counts, matching cells by `cell_id` (parsed fov + cellID). All 6,182 genes shared; 100% cell matching; correlation with nCount_RNA = 1.0000 for all TMAs.

| Dataset | Cells | Matched | Correlation with nCount_RNA |
|---------|-------|---------|---------------------------|
| TMA2_raw | 70,845 | 100% | 1.0000 |
| TMA4_raw | 90,602 | 100% | 1.0000 |
| TMA5_raw | 95,065 | 100% | 1.0000 |

| Config | Train | Eval | Purpose |
|--------|-------|------|---------|
| `raw_train.yaml` | TMA4,5 (raw) + census | TMA2 (raw) | Effect of real raw counts (compare vs baseline) |

### Data — TODO

- **LBCL atlas dataset**: Download from cellxgene, group-wise annotation generation, train un-aggregated.
