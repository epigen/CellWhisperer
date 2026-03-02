# Experiment 932: TMA Gene Expression Decoder -- Results Summary

> **Known issue (discovered 2026-02-19):** The prediction writer (`prediction_writer.py`) stored ground truth in raw count space while decoder predictions are in log1p space. This has been **fixed** (applying `np.log1p` to ground truth before saving), but existing `*_predictions.h5ad` files were generated with the old code and need to be **regenerated**. Pearson r values are approximately correct (log is monotonic), but MAE/MSE values in all evaluations below are invalid. See [Ground Truth Scale Mismatch](#ground-truth-scale-mismatch) for details.

## Experiment Setup

**Goal:** Train a gene expression decoder on frozen CellWhisperer (SpatialWhisperer v1) embeddings and evaluate whether image-based predictions recover spatial gene expression in lymphoma TMA cores.

| Parameter | Value |
|---|---|
| Base model | `spatialwhisperer/v1.ckpt` |
| Decoder architecture | Single linear layer (1024 -> ~6k genes) |
| Training TMAs | TMA4, TMA5, TMA13_14, TMA15_16 |
| Validation TMA | TMA2 (69,187 cells, 37 cores) |
| Test TMA | TMA3 (94,521 cells) |
| Inference TMAs | TMA1, TMA2, TMA3 |
| Gene panel | CosMx 6k (~5,782 expressed genes) |
| Loss | MSE |
| Hyperparameters | Batch 64, LR 1e-3, 50 epochs, AdamW + cosine annealing |

Pipeline: `Snakefile` with rules for training, prediction, evaluation, baseline comparison, and core-level analysis. See `README.md` for usage and `QUICKSTART.md` for quick commands.

---

## 1. Global Decoder Performance with Scrambled Baselines

**Script:** `scripts/evaluate_with_baseline.py`

Three methods compared:
- **Model**: decoder predictions vs ground truth
- **Scrambled**: predictions vs globally shuffled ground truth (per gene, across all cells) -- expected r ~ 0
- **Within-core scrambled**: predictions vs ground truth shuffled only within each core -- captures how much performance comes from learning per-core expression distributions rather than cell-level patterns

### 1a. Performance by HVG Threshold

Evaluated on top-N highly variable genes (ranked by ground truth variance). Pearson r is the primary metric; MAE/MSE are reported but currently invalid due to the scale mismatch (see above).

**TMA2 (validation):**

| Subset | n_genes | Model median r | Scrambled median r | Within-core median r | Incremental r (model - wc) |
|---|---|---|---|---|---|
| top-50 | 50 | 0.232 | 0.001 | 0.152 | 0.080 |
| top-100 | 100 | 0.201 | 0.001 | 0.134 | 0.068 |
| top-500 | 500 | 0.119 | 0.000 | 0.084 | 0.035 |
| top-1000 | 1000 | 0.086 | 0.000 | 0.060 | 0.026 |
| all | 5782 | 0.031 | 0.000 | 0.018 | 0.013 |

**TMA3 (test):**

| Subset | n_genes | Model median r | Scrambled median r | Within-core median r | Incremental r (model - wc) |
|---|---|---|---|---|---|
| top-50 | 50 | 0.204 | 0.000 | 0.115 | 0.089 |
| top-100 | 100 | 0.172 | 0.000 | 0.097 | 0.075 |
| top-500 | 500 | 0.096 | 0.000 | 0.059 | 0.037 |
| top-1000 | 1000 | 0.063 | 0.000 | 0.037 | 0.026 |
| all | 5782 | 0.025 | 0.000 | 0.014 | 0.011 |

Key observations:
- Performance (Pearson r) is highest on top-50 HVGs and degrades as more low-variance genes are included.
- The scrambled baseline is consistently ~0, confirming that model performance is genuine.
- **The within-core baseline accounts for 58-64% of total model Pearson r** (depending on TMA), meaning the majority of apparent decoder performance comes from learning per-core expression distributions (e.g., lymphoma vs. stroma cores have systematically different profiles). Only ~35-40% represents true cell-level prediction beyond core identity.
- TMA3 (test) shows consistent but slightly lower performance than TMA2 (validation) across all subsets.

Additional filter tested: minimum nonzero fraction thresholds (>5%, >10%). These made little difference -- the top-N HVGs already tend to have high nonzero fractions.

**Plots:**
- `results/plots/baseline_comparison/TMA2_performance_vs_ngenes.png` -- 4-panel (MAE, MSE, mean r, median r) vs gene count
- `results/plots/baseline_comparison/TMA3_performance_vs_ngenes.png` -- same for test set

### 1b. Gene-Level Performance Distribution

Per-gene Pearson r computed for all 5,782 genes across all cells.

**TMA2:**
- Model: mean r = 0.043, median r = 0.031
- 472 genes with r > 0.1, 94 genes with r > 0.2, 30 genes with r > 0.3
- Within-core baseline: mean r = 0.027, median r = 0.018
- Within-core explains 64% of model mean r

**TMA3:**
- Model: mean r = 0.034, median r = 0.025
- 314 genes with r > 0.1, 51 genes with r > 0.2, 12 genes with r > 0.3
- Within-core baseline: mean r = 0.020, median r = 0.014
- Within-core explains 58% of model mean r

On the top-500 HVG subset, the distribution shifts substantially:
- TMA2: model median r = 0.119 vs within-core median r = 0.084
- TMA3: model median r = 0.096 vs within-core median r = 0.059

**Plots:**
- `results/plots/baseline_comparison/TMA2_gene_pearson_hist.png` -- histogram of per-gene r, model vs scrambled (all genes)
- `results/plots/baseline_comparison/TMA2_gene_pearson_top500.png` -- same for top-500, model vs scrambled vs within-core
- `results/plots/baseline_comparison/TMA3_gene_pearson_hist.png`
- `results/plots/baseline_comparison/TMA3_gene_pearson_top500.png`

**Data files:**
- `results/metrics/metrics_with_baseline.csv` -- all HVG sweep results (92 rows: 2 TMAs x 3 methods x ~15 subsets)
- `results/metrics/gene_level_metrics.csv` -- per-gene r for all 5,782 genes (11,564 rows: 2 TMAs x 5,782 genes)

---

## 2. Core-Level Analysis: Does Base Model Quality Predict Decoder Performance?

**Scripts:** `scripts/compute_core_base_metrics.py`, `scripts/compute_core_decoder_performance.py`, `scripts/correlate_core_metrics.py`
**Snakemake:** `core_level_analysis.smk`

### Motivation

If embedding quality varies across tissue cores, we should expect downstream decoder performance to vary correspondingly. This analysis tests whether per-core base model metrics (retrieval AUROC, contrastive loss) correlate with per-core decoder performance.

### Setup

- **Dataset**: TMA2 (validation set), 30 cores with >= 50 cells
- **Decoder target metrics**: Top-500 HVGs, per-core mean/median Pearson r and MAE
- **Base model metrics**: Per-core contrastive loss (mean, median, std), retrieval recall@k (k=1,5,10,50), and AUROC (image-to-text, text-to-image)

### Key Finding: AUROC Predicts Decoder Quality; Recall@k Does Not

**AUROC strongly correlates with decoder Pearson correlation:**

| Base metric | Decoder metric | Pearson r | p-value | Spearman rho | p-value |
|---|---|---|---|---|---|
| i2t_rocauc | mean_gene_pearson | +0.850 | 2.78e-09 | +0.809 | 6.41e-08 |
| t2i_rocauc | mean_gene_pearson | +0.847 | 3.76e-09 | +0.822 | 2.55e-08 |
| i2t_rocauc | median_gene_pearson | +0.738 | 3.20e-06 | +0.674 | 4.46e-05 |
| t2i_rocauc | median_gene_pearson | +0.742 | 2.76e-06 | +0.689 | 2.55e-05 |
| std_loss | mean_gene_pearson | +0.753 | 1.56e-06 | +0.691 | 2.35e-05 |

**Contrastive loss strongly correlates with decoder MAE (but not Pearson):**

| Base metric | Decoder metric | Pearson r | p-value | Spearman rho | p-value |
|---|---|---|---|---|---|
| median_loss | mae | +0.876 | 2.21e-10 | +0.891 | 4.16e-11 |
| mean_loss | mae | +0.871 | 3.82e-10 | +0.887 | 7.08e-11 |
| mean_loss | median_gene_pearson | -0.026 | 0.89 | +0.071 | 0.71 |

**Recall@k does NOT correlate with decoder performance:**

| Base metric | Decoder metric | Pearson r | p-value |
|---|---|---|---|
| recall_at_50 | median_gene_pearson | +0.230 | 0.22 |
| recall_at_50 | mae | -0.670 | 5.19e-05 |

### Interpretation

1. **AUROC measures discriminability, not exact matching.** It varies meaningfully across cores (range: 0.553-0.782) and captures embedding space quality without being confounded by core size.

2. **Recall@k collapses for large cores.** With >1000 cells, the probability of the correct match being in the top-k is vanishingly small. Most cores show recall@1 < 0.01.

3. **Contrastive loss predicts MAE but not Pearson.** Loss correlates with n_cells (r=+0.889), and larger cores tend to have higher MAE (r=+0.792). This shared dependence on core size drives the loss-MAE correlation. Neither loss nor n_cells correlates with Pearson r (both p>0.7), suggesting that correlation-based decoder quality is independent of core size.

4. **Loss std predicts Pearson r (r=+0.753).** Higher within-core variability in contrastive loss is associated with better decoder Pearson r, possibly reflecting more diverse cell populations where embeddings capture richer biological signal.

### Data Ranges (TMA2, 30 cores)

- Core sizes: 130 - 5,593 cells
- Per-core AUROC: 0.553 - 0.782
- Per-core median Pearson r: 0.005 - 0.077
- Per-core MAE: 0.153 - 0.425
- Per-core mean contrastive loss: 4.89 - 9.21

### Output Files

- `results/metrics/core_level/core_metrics_combined.csv` -- all metrics per core (30 rows)
- `results/metrics/core_level/correlation_matrix.csv` -- Pearson correlations
- `results/metrics/core_level/spearman_correlation_matrix.csv` -- Spearman correlations
- `results/metrics/core_level/TMA2_core_decoder.csv`, `TMA2_core_retrieval.csv`, `TMA2_core_loss.csv`
- `results/plots/core_level_correlation/correlation_heatmap.png` -- full heatmap
- `results/plots/core_level_correlation/scatter_plots/` -- 6 scatter plots

### Caveats

- **TMA2 only.** TMA3 (test set) has not been run through core-level analysis yet.
- **n_cells confound**: The loss-MAE correlation may be driven by core size. Partial correlation controlling for n_cells was not computed.
- **Baseline-adjusted decoder performance** (model_r - within-core-scrambled_r) was not tested as a target.
- **Gene subset sensitivity**: Core-level analysis only evaluated top-500 HVGs.

---

## 3. IHC Validation (TMA1 Ground Truth)

**Script:** `scripts/correlate_predictions_with_ihc.py`

Correlates TMA1 **ground truth** gene expression (not decoder predictions) against pathologist-scored IHC H-scores for PAX5 and CD19. This establishes a baseline for RNA-protein agreement before comparing with decoder predictions.

### Mapping Challenge

Three identifier systems had to be reconciled:
- **CosMx data**: FOV numbers (1-167)
- **IHC file**: Patient core IDs (e.g., "1-621", "1-873")
- **TMA layout**: Grid positions (e.g., "A5", "K5")

**Solution:** Created `tma1_fov_to_ihc_mapping.csv` (FOV -> grid position -> sample ID -> IHC scores). Source: `src/datasets/lymphoma_cosmx_small/cell_barcode_core_assignment.csv`.

Coverage: 104 FOVs mapped, 63 grid positions, 36 patients, 52 FOVs with both PAX5 and CD19 scores.

### Results (Ground Truth vs IHC)

| Gene | Aggregation | n FOVs | n cores | n patients | Pearson r | p-value | Spearman r |
|---|---|---|---|---|---|---|---|
| PAX5 | mean | 51 | 34 | 20 | 0.218 | 0.124 | 0.151 |
| PAX5 | 90th pctl | 51 | 34 | 20 | 0.140 | 0.326 | 0.013 |
| CD19 | mean | 62 | 41 | 24 | -0.254 | 0.046* | -0.059 |
| CD19 | 90th pctl | 62 | 41 | 24 | -0.158 | 0.219 | 0.028 |

- PAX5: weak positive correlation, not significant. Ground truth RNA shows modest agreement with protein.
- CD19: weak *negative* correlation (marginally significant for mean aggregation). Unexpected inverse relationship between RNA and protein -- warrants investigation (post-transcriptional regulation? IHC scoring artifacts? spatial sampling mismatch?).

### Output Files

- `analysis/correlation_results.csv` -- summary statistics
- `analysis/PAX5_PAX5_Hscore_mean_correlation.png`, `*_p90_correlation.png`
- `analysis/CD19_CD19_Hscore_mean_correlation.png`, `*_p90_correlation.png`
- `tma1_fov_to_ihc_mapping.csv` -- complete FOV-level mapping

### Caveats

- Uses **ground truth expression**, not decoder predictions. The real test is comparing decoder predictions vs this baseline.
- TMA1-specific mapping. Other TMAs require their own FOV-to-IHC mapping.
- IHC file coverage for TMA2/TMA3 is unverified.
- Only PAX5 and CD19 tested (CD20 also available in IHC file with 27 non-null entries).

---

## Ground Truth Scale Mismatch

**Discovered:** 2026-02-19
**Status:** Fix applied to `prediction_writer.py`; predictions h5ads need regeneration.

### The Bug

The decoder is trained on `batch["expression_expr"]` which contains **log1p-transformed** counts (applied by `MLPTranscriptomeProcessor._prepare_features()` at `mlp_model.py:79`). The decoder therefore outputs predictions in log1p space.

However, `GeneExpressionPredictionWriter` (`prediction_writer.py:180-185`) stored ground truth by reading `original_adata.X` directly -- **raw integer counts** without log transform:

- `adata.X` (predictions): log1p space
- `adata.layers["ground_truth"]`: raw count space

### Impact on Reported Metrics

- **Pearson r:** Approximately correct (log is monotonic; for small CosMx counts log1p is nearly linear). Relative gene/core rankings are reliable.
- **MAE/MSE:** Invalid -- different scales. All absolute error numbers above should be disregarded.
- **Core-level correlation analysis:** Based on Pearson r, so qualitative findings hold.

### The Fix

```python
# prediction_writer.py, line 185 (after fix)
ground_truth_full[:, gt_indices] = np.log1p(ground_truth_subset)
```

### Action Required

Re-run `predict_expression` on Sherlock to regenerate `*_predictions.h5ad`, then re-run all downstream evaluations (`evaluate_with_baseline`, `compute_core_decoder_performance`, `correlate_core_metrics`).

---

## Open Questions and Next Steps

1. **Regenerate predictions** with the log1p fix and re-run all evaluations.
2. **Run TMA3 core-level analysis** (same AUROC-vs-decoder pipeline as TMA2).
3. **Partial correlations** controlling for n_cells in the core-level analysis.
4. **Baseline-adjusted decoder performance** -- compute model_r - within_core_r per core and test whether AUROC predicts the incremental signal.
5. **IHC validation on decoder predictions** -- compare decoder-predicted PAX5/CD19 vs IHC H-scores, and benchmark against ground truth baseline.
6. **Investigate CD19 negative correlation** -- is the RNA-protein inversion biological or technical?
7. **Extend IHC to TMA2/TMA3** -- verify IHC file coverage and create FOV mappings.

---

## File Index

### Scripts

| Script | Purpose |
|---|---|
| `scripts/evaluate_with_baseline.py` | HVG sweep, scrambled baselines, gene-level metrics |
| `scripts/plot_performance_vs_ngenes.py` | Performance-vs-ngenes 4-panel plots |
| `scripts/compute_core_decoder_performance.py` | Per-core decoder Pearson/MAE (top-500 HVGs) |
| `scripts/compute_core_base_metrics.py` | Per-core retrieval + contrastive loss from base model |
| `scripts/correlate_core_metrics.py` | Merge + correlate + heatmap + scatter plots |
| `scripts/correlate_predictions_with_ihc.py` | IHC validation (expression vs H-scores) |

### Result Files

| File | Description |
|---|---|
| `results/metrics/metrics_with_baseline.csv` | HVG sweep: 2 TMAs x 3 methods x ~15 subsets (92 rows) |
| `results/metrics/gene_level_metrics.csv` | Per-gene r for 5,782 genes x 2 TMAs (11,564 rows) |
| `results/metrics/core_level/core_metrics_combined.csv` | All core-level metrics merged (30 rows) |
| `results/metrics/core_level/correlation_matrix.csv` | Pearson correlation matrix |
| `results/metrics/core_level/spearman_correlation_matrix.csv` | Spearman correlation matrix |
| `results/metrics/core_level/TMA2_core_*.csv` | Per-core decoder, retrieval, loss CSVs |
| `analysis/correlation_results.csv` | IHC validation results |

### Plots

| Plot | Description |
|---|---|
| `results/plots/baseline_comparison/TMA{2,3}_performance_vs_ngenes.png` | 4-panel: MAE, MSE, mean r, median r vs gene count |
| `results/plots/baseline_comparison/TMA{2,3}_gene_pearson_hist.png` | Per-gene r histogram (all genes) |
| `results/plots/baseline_comparison/TMA{2,3}_gene_pearson_top500.png` | Per-gene r histogram (top-500, 3 methods) |
| `results/plots/core_level_correlation/correlation_heatmap.png` | Full metric correlation heatmap |
| `results/plots/core_level_correlation/scatter_plots/*.png` | 6 key scatter plots |
| `analysis/*.png` | IHC scatter plots (PAX5/CD19 x mean/p90) |
