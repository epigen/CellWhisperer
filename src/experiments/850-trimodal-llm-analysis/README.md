# Trimodal Foundation Model Analysis

Analysis pipeline for evaluating **trimodal** mapping, a model that aligns histopathology images, transcriptomics, and domain text through contrastive learning.

## Project Objective

This codebase performs per-class and retrieval-level analysis to understand **what drives improved (or decreased) prediction performance in trimodal vs bimodal learning** across three modality pairs:

1. **Image ↔ Text**: Histopathology patches aligned with pathology descriptions
2. **Transcriptome ↔ Image**: Gene expression profiles aligned with spatial tissue images  
3. **Transcriptome ↔ Text**: Gene expression profiles aligned with cell-type descriptions

### Research Questions
- **Biological factors**: Do certain diseases benefit more from trimodal learning than cell types?
- **Technical factors**: Does training data volume, gene count, or annotation quality affect performance?
- **High-CLIP cohorts**: What biological or technical patterns emerge in strongly aligned cross-modal pairs?

---

## Expected Data Structure

The analysis expects CSV files organized under `docs/per_class_analysis/` following this structure:

```
docs/per_class_analysis/
├── image-text/
│   ├── musk_per_class_analysis_seed0.csv          # Per-class F1 scores (image-text)
│   ├── quilt1m_per_class_analysis_retrieval.csv   # Retrieval CLIP scores (image-text)
│   └── quilt1m_sample_metadata.csv                # Patch metadata (sample_id, annotations, dataset)
├── transcriptome-image/
│   ├── hest1k_per_class_analysis.csv              # Per-class F1 scores (transcriptome-image)
│   ├── hest1k_per_class_analysis_retrieval.csv    # Retrieval CLIP scores (transcriptome-image)
│   └── hest1k_sample_metadata.csv                 # Spatial metadata (gene counts, coordinates)
└── transcriptome-text/
    ├── human_disease_per_class_analysis.csv       # Per-class F1 scores (diseases)
    ├── cellxgene_census_archs4_geo_per_class_analysis_retrieval.csv  # Retrieval CLIP scores
    ├── cellxgene_census_sample_metadata.csv       # Cell-type metadata (cell_type, tissue, organ)
    └── archs4_geo_sample_metadata.csv             # Bulk RNA-seq metadata (biosample_title, ontology terms)
```

### Required CSV Columns

**Per-class analysis files** must include:
- `class_label` or equivalent (orig_ids, class, class_type, class_name)
- `bimodal_matching`: F1 score for bimodal model
- `trimodal`: F1 score for trimodal model
- `improvement`: Absolute improvement (trimodal - bimodal)
- `relative_improvement`: Relative improvement ((trimodal - bimodal) / bimodal)
- `dataset`: Dataset identifier

**Retrieval analysis files** must include:
- `orig_ids`: Unique identifier for each sample/patch
- `avg_clip_score`: Average CLIP similarity score for retrieval

**Metadata files** should include:
- `sample_id` or `orig_ids`: Join key
- Domain-specific fields (e.g., `organ_site`, `stain_type`, `tissue_type`, `cell_type`, `gene_counts`)

---

## Analysis Methodology

### 1. Per-Class Performance Analysis (`analysis/per_class_analysis.py`)

**Purpose**: Aggregate per-class F1 scores across all modality pairs to identify where trimodal learning excels or underperforms.

**Process**:
1. **Load & normalize**: Reads CSVs from all three modality pairs and harmonizes column names
2. **Compute statistics**:
   - Overall improvement (mean, median, std) per modality
   - Top 10 classes with largest improvements per modality
   - Bottom 10 classes with largest declines per modality
   - Relative improvement summaries
3. **Generate visualizations**:
   - Mean improvement bar charts
   - Per-class improvement distributions (KDE plots)
   - Improvement scatter plots by modality
   - Modality-specific histograms
   - Top/bottom 10 class bar charts

**Outputs** → `analysis/outputs/`:
- `per_class_combined.csv`: All per-class metrics in unified format
- `overall.csv`: Summary statistics per modality
- `top_improvements.csv`: Top 10 improvements per modality
- `worst_declines.csv`: Worst 10 declines per modality
- `relative_summary.csv`: Relative improvement statistics
- `summary.json`: JSON export of all summary tables
- `plots/*.png`: Publication-ready figures

---

### 2. Retrieval Analysis (`analysis/retrieval_analysis.py`)

**Purpose**: Join retrieval CLIP scores with sample metadata to identify high-scoring cross-modal pairs and aggregate by biological/technical covariates.

**Process**:
1. **Load & merge**: Joins retrieval CSVs with metadata for each modality pair
2. **Flag high-CLIP pairs**: Uses 95th percentile threshold (configurable via `--quantile`) to identify strongly aligned pairs
3. **Aggregate summaries**:
   - **Image-Text**: Group by `sample_id` and `dataset`; compute high-CLIP fractions, mean scores, annotation examples
   - **Transcriptome-Image**: Group by `sample_id`; include gene count statistics and spatial metadata
   - **Transcriptome-Text**: Group by `dataset` and `cell_type`; split ARCHS4/GEO vs CellxGene Census cohorts
4. **Export share payloads**: Create `share_payload.json` with top-N entries per modality and embedded prompts for downstream LLM analysis

**Outputs** → `analysis/outputs/retrieval/`:
- `image-text/quilt1m_sample_summary.csv`: High-CLIP samples (image-text)
- `image-text/quilt1m_sample_summary_share.csv`: Share-ready summary with annotation examples
- `transcriptome-image/hest1k_sample_summary.csv`: High-CLIP samples (transcriptome-image)
- `transcriptome-image/hest1k_sample_summary_share.csv`: Share-ready summary with gene counts
- `transcriptome-text/transcriptome_text_cell_type_summary.csv`: High-CLIP cell types
- `transcriptome-text/transcriptome_text_dataset_summary.csv`: High-CLIP datasets
- `share_payload.json`: Top-N entries bundled with prompts for LLM analysis

---

### 3. HTML Report Generation (`analysis/generate_report.py`)

**Purpose**: Create a self-contained HTML report with embedded figures, collapsible tables, and interactive visualizations.

**Features**:
- **Performance Snapshot**: Summary cards with class counts, mean/median improvements per modality
- **Retrieval Overview**: High-CLIP sample counts and thresholds per modality
- **Key Findings**: Biological insights (e.g., necrosis patterns, immune cell types, spatial richness)
- **Interactive Figures**: Click-to-expand modal viewer for all plots
- **Scatter Plots**: CLIP score vs gene counts, high-CLIP fractions
- **Downloadable Tables**: CSV downloads for all summary tables

**Output** → `analysis/outputs/per_class_summary.html`

---

## Outputs Summary

| Category | Files | Description |
|----------|-------|-------------|
| **Combined Data** | `per_class_combined.csv` | All per-class metrics in unified schema |
| **Summary Stats** | `overall.csv`, `relative_summary.csv`, `summary.json` | Aggregated statistics per modality |
| **Top/Bottom Classes** | `top_improvements.csv`, `worst_declines.csv` | Extreme performers per modality |
| **Plots** | `plots/*.png` (8 figures) | Bar charts, histograms, KDE plots, scatter plots |
| **Retrieval Summaries** | `retrieval/*/*.csv` (12 files) | High-CLIP samples grouped by metadata |
| **Share Payload** | `share_payload.json` | Top-N entries with LLM prompts for each modality |
| **HTML Report** | `per_class_summary.html` | Interactive dashboard with all results |

---

## Re-run Commands

### Quick Start (Default Paths)

**One-command pipeline** (runs all three scripts):
```bash
./run_analysis.sh
```

Or run scripts individually:
```bash
# 1. Run per-class analysis (generates stats + plots)
uv run python analysis/per_class_analysis.py

# 2. Run retrieval analysis (generates high-CLIP summaries + share payload)
uv run python analysis/retrieval_analysis.py

# 3. Generate HTML report (combines all outputs)
uv run python analysis/generate_report.py
```

### Advanced Usage

**Per-class analysis with custom paths**:
```bash
uv run python analysis/per_class_analysis.py \
  --data-dir /path/to/per_class_analysis \
  --output-dir /path/to/outputs
```

**Retrieval analysis with custom threshold**:
```bash
uv run python analysis/retrieval_analysis.py \
  --quantile 0.90 \              # Use 90th percentile instead of 95th
  --top-n 50 \                   # Include top 50 entries in share payload
  --no-share-payload             # Skip share_payload.json generation
```

**Standalone HTML report generation**:
```bash
uv run python analysis/generate_report.py
# Expects outputs already present in analysis/outputs/
```

---

## Environment Setup

This project uses **uv** for dependency management.

### Installation
```bash
# Install dependencies
uv sync

# Or manually add packages
uv add pandas matplotlib seaborn
```

### Requirements
- Python 3.12+
- pandas ≥2.3.3
- matplotlib ≥3.10.6
- seaborn ≥0.13.2

See `pyproject.toml` for full dependency list.

---

## Key Findings (Oct 2025)

### Per-Class Performance
- **Image-Text**: 24 classes, mean improvement +0.015
- **Transcriptome-Text**: 218 classes, mean improvement +0.028
- **Transcriptome-Image**: 48 classes, mean improvement +0.008

### Retrieval Highlights
1. **Necrosis & Hyaline Membranes**: Image-Text cohorts dominated by glioblastoma necrosis and diffuse alveolar damage patterns
2. **Spatial Richness**: HEST1K samples with high gene counts (>5000) show strongest Transcriptome-Image alignment
3. **Immune & CNS Cell Types**: Transcriptome-Text peaks for germinal-center B cells, memory B cells, neurons, and oligodendrocytes
4. **Curated Atlases Excel**: CellxGene Census far outperforms ARCHS4/GEO in CLIP alignment

---

## Outstanding Data Needs

To enable deeper biological interpretation, we need:

1. **Quilt1M metadata**: `sample_id → organ_site, stain_type, diagnosis`
2. **HEST1K metadata**: `sample_id → tissue_type, sequencing_platform, stain`
3. **ARCHS4/GEO metadata**: `accession → cell_line, treatment, assay_type`

Once available, rerun the retrieval pipeline to enrich summaries:
```bash
uv run python analysis/retrieval_analysis.py --top-n 30
uv run python analysis/generate_report.py
```

---

## Project Structure

```
trimodal/
├── analysis/
│   ├── per_class_analysis.py      # Per-class performance aggregation
│   ├── retrieval_analysis.py      # High-CLIP retrieval analysis
│   ├── generate_report.py         # HTML report generation
│   ├── outputs/                   # Generated outputs
│   └── README.md                  # Analysis-specific documentation
├── docs/
│   ├── per_class_analysis/        # Input CSV files (per-class + retrieval + metadata)
│   └── Carambola_ICLR_2025.pdf    # Manuscript draft
├── memory/
│   ├── chat_logs/                 # Session logs
│   ├── chat_summary/              # Session summaries
│   └── README.md                  # Memory management guidelines
├── pyproject.toml                 # UV project configuration
└── README.md                      # This file
```

---

## Related Issues

- Analysis of trimodal benefits [#850](https://github.com/epigen/cellwhisperer_private/issues/850)

---

**Last Updated**: October 15, 2025

