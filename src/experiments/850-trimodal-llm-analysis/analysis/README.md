# Analysis Scripts

This directory contains the trimodal vs bimodal analysis pipeline. See the [main README](../README.md) for complete documentation.

## Scripts Overview

### `per_class_analysis.py`
Aggregates per-class F1 scores across all modality pairs (image-text, transcriptome-text, transcriptome-image) to compute improvement statistics and generate visualizations.

**Quick run**:
```bash
uv run python analysis/per_class_analysis.py
```

**Options**:
- `--data-dir`: Path to input CSVs (default: `docs/per_class_analysis`)
- `--output-dir`: Path for outputs (default: `analysis/outputs`)

---

### `retrieval_analysis.py`
Joins retrieval CLIP scores with sample metadata to identify high-scoring cross-modal pairs and aggregate by biological/technical covariates.

**Quick run**:
```bash
uv run python analysis/retrieval_analysis.py
```

**Options**:
- `--data-root`: Base directory for retrieval CSVs and metadata (default: `docs/per_class_analysis`)
- `--output-root`: Directory for merged tables (default: `analysis/outputs/retrieval`)
- `--quantile`: Quantile threshold for high-CLIP pairs (default: 0.95)
- `--top-n`: Number of top entries in share payload (default: 30)
- `--no-share-payload`: Skip `share_payload.json` generation

---

### `generate_report.py`
Creates a self-contained HTML report with embedded figures, interactive tables, and retrieval visualizations.

**Quick run**:
```bash
uv run python analysis/generate_report.py
```

**Note**: Expects outputs already present in `analysis/outputs/`. Run the previous two scripts first.

---

## Complete Pipeline

Run all three scripts in sequence:

```bash
uv run python analysis/per_class_analysis.py && \
uv run python analysis/retrieval_analysis.py && \
uv run python analysis/generate_report.py
```

View results: Open `analysis/outputs/per_class_summary.html` in your browser.

---

## Outputs

All outputs are written to `analysis/outputs/`:

- **Per-class CSVs**: `per_class_combined.csv`, `overall.csv`, `top_improvements.csv`, `worst_declines.csv`, `relative_summary.csv`
- **Plots**: `plots/*.png` (8 publication-ready figures)
- **Retrieval summaries**: `retrieval/*/*.csv` (12 files grouped by modality)
- **Share payload**: `retrieval/share_payload.json` (top-N entries with LLM prompts)
- **HTML report**: `per_class_summary.html` (interactive dashboard)

