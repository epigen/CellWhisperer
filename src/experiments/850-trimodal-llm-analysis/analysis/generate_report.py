#!/usr/bin/env python3
"""Generate a self-contained HTML report summarizing per-class analysis results."""

from __future__ import annotations

import base64
import io
import json
import re
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORT_PATH = OUTPUT_DIR / "per_class_summary.html"


def encode_png(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def df_to_html(
    df: pd.DataFrame,
    caption: str | None = None,
    max_rows: int | None = None,
    download: str | None = None,
) -> str:
    table_df = df.copy()
    truncated = False
    if max_rows is not None and len(table_df) > max_rows:
        table_df = table_df.head(max_rows)
        truncated = True
    html = table_df.to_html(index=False, escape=False, classes="table table-sm table-striped")
    caption_html = f"<caption>{caption}</caption>" if caption else ""
    download_html = (
        f"<p class=\"download-link\"><a href=\"{download}\" download>Download CSV</a></p>"
        if download
        else ""
    )
    note_html = (
        f"<p class=\"note\">Showing first {max_rows} rows of {len(df)} (download for full table).</p>"
        if truncated
        else ""
    )
    return f"<div class=\"table-wrapper\">{caption_html}{download_html}{html}{note_html}</div>"


def build_summary(overall: pd.DataFrame) -> str:
    items = []
    for _, row in overall.iterrows():
        modality = row["modality"]
        count = int(row["count"])
        mean = row["mean"]
        median = row["median"]
        items.append(
            f"<li><strong>{modality}</strong>: {count} classes, mean Δ={mean:.3f}, median Δ={median:.3f}</li>"
        )
    return "<ul>" + "".join(items) + "</ul>"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def collapsible_block(
    title: str,
    body: str,
    *,
    open_default: bool = False,
    download: str | None = None,
) -> str:
    open_attr = " open" if open_default else ""
    download_html = (
        f"<p class=\"download-link\"><a href=\"{download}\" download>Download CSV</a></p>"
        if download
        else ""
    )
    return (
        f"<details id=\"{slugify(title)}\" class=\"collapsible\"{open_attr}>"
        f"<summary>{title}</summary>"
        f"{download_html}{body}</details>"
    )


def build_figures_section(
    images: dict[str, str],
    *,
    section_id: str,
    heading: str,
) -> str:
    if not images:
        return ""

    preferred_order = [
        "bottom10_declines",
        "top10_improvements",
        "mean_improvement_by_modality",
        "histogram_image-text",
        "histogram_transcriptome-text",
        "histogram_transcriptome-image",
    ]

    ordered_keys: list[str] = []
    for key in preferred_order:
        if key in images:
            ordered_keys.append(key)
    for key in sorted(images.keys()):
        if key not in ordered_keys:
            ordered_keys.append(key)

    cards = []
    for key in ordered_keys:
        data_uri = images[key]
        title = key.replace("_", " ").title()
        cards.append(
            dedent(
                f"""
                <div class=\"figure-card\">
                  <img src="{data_uri}" alt="{title}" data-full="{data_uri}" data-title="{title}" />
                  <div class=\"figure-caption\">{title}</div>
                  <button type=\"button\" class=\"expand-btn\" data-full="{data_uri}" data-title="{title}">View larger</button>
                </div>
                """
            )
        )

    return (
        f"<section id=\"{section_id}\" class=\"figure-section\">"
        f"<h3>{heading}</h3>"
        "<div class=\"figure-grid\">"
        + "".join(cards)
        + "</div></section>"
    )


def build_nav() -> str:
    links = [
        ("Summary", "summary"),
        ("Performance Figures", "performance-figures"),
        ("Retrieval Visuals", "retrieval-figures"),
        ("Performance Tables", "tables"),
        ("Retrieval Tables", "retrieval-tables"),
        ("JSON", "json"),
    ]
    items = "".join(f"<a href=\"#{anchor}\">{label}</a>" for label, anchor in links)
    return f"<nav class=\"top-nav\">{items}</nav>"


def generate_retrieval_figure(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str) -> str:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[x], df[y], alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=160)
    plt.close()
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall = pd.read_csv(OUTPUT_DIR / "overall.csv")
    relative = pd.read_csv(OUTPUT_DIR / "relative_summary.csv")
    top = pd.read_csv(OUTPUT_DIR / "top_improvements.csv")
    bottom = pd.read_csv(OUTPUT_DIR / "worst_declines.csv")
    combined = pd.read_csv(OUTPUT_DIR / "per_class_combined.csv")

    counts_by_dataset = (
        combined.groupby(["modality", "dataset"]).size().reset_index(name="count")
    )
    relative_json = json.dumps(json.loads(relative.to_json(orient="records")), indent=2)

    images: dict[str, str] = {}
    if PLOTS_DIR.exists():
        for path in sorted(PLOTS_DIR.glob("*.png")):
            images[path.stem] = encode_png(path)

    # Load retrieval summaries if they exist
    retrieval_dir = OUTPUT_DIR / "retrieval"
    retrieval_tables: dict[str, pd.DataFrame] = {}
    retrieval_downloads: dict[str, str] = {}

    if retrieval_dir.exists():
        retrieval_configs = {
            "Image-Text High-CLIP Samples": (
                retrieval_dir / "image-text" / "quilt1m_sample_summary.csv",
                "retrieval/image-text/quilt1m_sample_summary.csv",
            ),
            "Transcriptome-Image High-CLIP Samples": (
                retrieval_dir / "transcriptome-image" / "hest1k_sample_summary.csv",
                "retrieval/transcriptome-image/hest1k_sample_summary.csv",
            ),
            "Transcriptome-Text High-CLIP Cell Types": (
                retrieval_dir / "transcriptome-text" / "transcriptome_text_cell_type_summary.csv",
                "retrieval/transcriptome-text/transcriptome_text_cell_type_summary.csv",
            ),
        }

        for label, (path, download) in retrieval_configs.items():
            if path.exists():
                retrieval_tables[label] = pd.read_csv(path)
                retrieval_downloads[label] = download
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary_html = build_summary(overall)

    summary_cards: list[str] = []
    for _, row in overall.iterrows():
        modality_title = str(row["modality"]).replace("-", " ").title()
        summary_cards.append(
            dedent(
                f"""
                <div class="summary-card">
                  <h4>{modality_title}</h4>
                  <p><span class="metric">{int(row['count'])}</span> classes</p>
                  <p>Mean improvement <span class="metric">{row['mean']:.3f}</span></p>
                  <p>Median improvement <span class="metric">{row['median']:.3f}</span></p>
                </div>
                """
            ).strip()
        )

    summary_box_html = (
        "<div class=\"summary-box\"><h3>Performance Snapshot</h3><div class=\"summary-grid\">"
        + "".join(summary_cards)
        + "</div></div>"
    ) if summary_cards else ""

    metadata_links = (
        "<div class=\"metadata-links\">"
        "<strong>Source Data:</strong> "
        "<a href=\"../docs/per_class_analysis/transcriptome-text/human_disease_per_class_analysis.csv\">Transcriptome-Text</a> | "
        "<a href=\"../docs/per_class_analysis/transcriptome-image/hest1k_per_class_analysis.csv\">Transcriptome-Image</a> | "
        "<a href=\"../docs/per_class_analysis/image-text/musk_per_class_analysis_seed0.csv\">Image-Text</a>"
        "</div>"
    )

    retrieval_overview_html = ""
    if retrieval_tables:
        def _threshold_from(path: Path) -> float | None:
            threshold_path = path.parent / "threshold.txt"
            if threshold_path.exists():
                try:
                    return float(threshold_path.read_text().strip())
                except ValueError:
                    return None
            return None

        image_threshold = _threshold_from(retrieval_dir / "image-text" / "quilt1m_sample_summary.csv")
        timg_threshold = _threshold_from(retrieval_dir / "transcriptome-image" / "hest1k_sample_summary.csv")
        ttxt_threshold = _threshold_from(
            retrieval_dir / "transcriptome-text" / "transcriptome_text_cell_type_summary.csv"
        )

        def _fmt_threshold(value: float | None) -> str:
            return f"{value:.2f}" if value is not None else "–"

        retrieval_overview_html = dedent(
            f"""
            <div class="summary-box retrieval-box">
              <h3>Retrieval High-CLIP Overview</h3>
              <ul>
                <li><strong>Image-Text</strong>: {len(retrieval_tables.get('Image-Text High-CLIP Samples', []))} samples (threshold {_fmt_threshold(image_threshold)})</li>
                <li><strong>Transcriptome-Image</strong>: {len(retrieval_tables.get('Transcriptome-Image High-CLIP Samples', []))} samples (threshold {_fmt_threshold(timg_threshold)})</li>
                <li><strong>Transcriptome-Text</strong>: {len(retrieval_tables.get('Transcriptome-Text High-CLIP Cell Types', []))} cell types (threshold {_fmt_threshold(ttxt_threshold)})</li>
              </ul>
            </div>
            """
        )

    key_findings = dedent(
        """
        <div class="summary-box retrieval-box">
          <h3>Key Retrieval Findings</h3>
          <ul>
            <li><strong>Glioblastoma necrosis & diffuse alveolar damage</strong>: Image↔Text cohorts are dominated by patches showing necrotic astrocytic tumours and eosinophilic hyaline membranes, explaining strong alignment with pathology descriptions.</li>
            <li><strong>Spatial transcriptomics richness</strong>: HEST1K samples with high gene counts and dense tissue coverage (e.g., TENX/TENX152) deliver the strongest Transcriptome↔Image matches.</li>
            <li><strong>Immune and CNS cell types</strong>: Transcriptome↔Text alignment peaks for germinal-centre B cells, memory B cells, neurons, and oligodendrocytes, indicating that CLIP captures both immunoglobulin and neural marker programs.</li>
            <li><strong>Curated cell atlases excel</strong>: cellxgene_census far outperforms archs4_geo in Text alignment, underscoring the value of standardized single-cell metadata.</li>
          </ul>
        </div>
        """
    ) if retrieval_tables else ""

    summary_section = (
        "<section id=\"summary\">"
        f"<h2>Summary</h2><p class=\"details\">Report generated on {generated_at}.</p>"
        + metadata_links
        + summary_box_html
        + retrieval_overview_html
        + key_findings
        + df_to_html(relative, caption="Relative Improvement Summary", download="relative_summary.csv")
        + "</section>"
    )

    figures_section = build_figures_section(
        {k: v for k, v in images.items()},
        section_id="performance-figures",
        heading="Test Performance Visuals",
    )

    retrieval_figures_html = ""
    if retrieval_tables:
        figure_cards = []

        image_text_df = retrieval_tables.get("Image-Text High-CLIP Samples")
        if image_text_df is not None and not image_text_df.empty:
            scatter_uri = generate_retrieval_figure(
                image_text_df,
                x="mean_clip_score",
                y="high_clip_fraction",
                title="Image-Text: Mean CLIP vs High-Score Fraction",
                xlabel="Mean CLIP Score",
                ylabel="High-CLIP Fraction",
            )
            figure_cards.append(
                dedent(
                    f"""
                    <div class="figure-card">
                      <img src="{scatter_uri}" alt="Image-Text Scatter" data-full="{scatter_uri}" data-title="Image-Text: Mean CLIP vs High-Score Fraction" />
                      <div class="figure-caption">Image-Text: Mean CLIP vs High-Score Fraction</div>
                      <button type="button" class="expand-btn" data-full="{scatter_uri}" data-title="Image-Text: Mean CLIP vs High-Score Fraction">View larger</button>
                    </div>
                    """
                )
            )

        transcriptome_image_df = retrieval_tables.get("Transcriptome-Image High-CLIP Samples")
        if transcriptome_image_df is not None and not transcriptome_image_df.empty:
            scatter_uri = generate_retrieval_figure(
                transcriptome_image_df,
                x="mean_gene_counts",
                y="high_clip_fraction",
                title="Transcriptome-Image: Gene Counts vs High-Score Fraction",
                xlabel="Mean Gene Counts",
                ylabel="High-CLIP Fraction",
            )
            figure_cards.append(
                dedent(
                    f"""
                    <div class="figure-card">
                      <img src="{scatter_uri}" alt="Transcriptome-Image Scatter" data-full="{scatter_uri}" data-title="Transcriptome-Image: Gene Counts vs High-Score Fraction" />
                      <div class="figure-caption">Transcriptome-Image: Gene Counts vs High-Score Fraction</div>
                      <button type="button" class="expand-btn" data-full="{scatter_uri}" data-title="Transcriptome-Image: Gene Counts vs High-Score Fraction">View larger</button>
                    </div>
                    """
                )
            )

        transcriptome_text_df = retrieval_tables.get("Transcriptome-Text High-CLIP Cell Types")
        if transcriptome_text_df is not None and not transcriptome_text_df.empty:
            scatter_uri = generate_retrieval_figure(
                transcriptome_text_df,
                x="high_clip_count",
                y="mean_clip_score",
                title="Transcriptome-Text: High-Clip Count vs Mean Score",
                xlabel="High-Clip Count",
                ylabel="Mean CLIP Score",
            )
            figure_cards.append(
                dedent(
                    f"""
                    <div class="figure-card">
                      <img src="{scatter_uri}" alt="Transcriptome-Text Scatter" data-full="{scatter_uri}" data-title="Transcriptome-Text: High-Clip Count vs Mean Score" />
                      <div class="figure-caption">Transcriptome-Text: High-Clip Count vs Mean Score</div>
                      <button type="button" class="expand-btn" data-full="{scatter_uri}" data-title="Transcriptome-Text: High-Clip Count vs Mean Score">View larger</button>
                    </div>
                    """
                )
            )

        if figure_cards:
            retrieval_figures_html = (
                "<section id=\"retrieval-figures\">"
                "<h2>Retrieval Visualizations</h2>"
                "<div class=\"figure-grid\">"
                + "".join(figure_cards)
                + "</div></section>"
            )

    tables_section = (
        "<section id=\"tables\">"
        "<h2>Performance Tables</h2>"
        + collapsible_block(
            "Counts by Modality and Dataset",
            df_to_html(counts_by_dataset, caption="Counts by Modality and Dataset", download="per_class_combined.csv"),
            open_default=True,
            download="per_class_combined.csv",
        )
        + collapsible_block(
            "Top Per-Modality Improvements",
            df_to_html(top, caption="Top Improvements", max_rows=30, download="top_improvements.csv"),
            download="top_improvements.csv",
        )
        + collapsible_block(
            "Worst Per-Modality Declines",
            df_to_html(bottom, caption="Worst Declines", max_rows=30, download="worst_declines.csv"),
            download="worst_declines.csv",
        )
        + collapsible_block(
            "Per-Class Metrics (sorted)",
            df_to_html(
                combined.sort_values(
                    ["modality", "dataset", "improvement"],
                    ascending=[True, True, False],
                ),
                caption="Per-Class Metrics",
                max_rows=200,
                download="per_class_combined.csv",
            ),
            download="per_class_combined.csv",
        )
        + "</section>"
    )

    if retrieval_tables:
        retrieval_blocks = []
        for label, table in retrieval_tables.items():
            retrieval_blocks.append(
                collapsible_block(
                    label,
                    df_to_html(
                        table,
                        caption=f"{label} (top rows)",
                        max_rows=25,
                        download=retrieval_downloads.get(label),
                    ),
                    download=retrieval_downloads.get(label),
                )
            )
        retrieval_tables_section = (
            "<section id=\"retrieval-tables\">"
            "<h2>Retrieval Tables</h2>"
            + "".join(retrieval_blocks)
            + "</section>"
        )
    else:
        retrieval_tables_section = ""

    json_section = (
        "<section id=\"json\">"
        "<h2>Relative Improvement JSON</h2>"
        "<details class=\"collapsible\" open>"
        "<summary>View JSON payload</summary>"
        f"<pre>{relative_json}</pre>"
        "</details></section>"
    )

    html = dedent(
        f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>Per-Class Trimodal Analysis Summary</title>
          <style>
            body {{ font-family: Arial, sans-serif; margin: 2rem; color: #222; }}
            .top-nav {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
            .top-nav a {{ text-decoration: none; color: #0366d6; font-weight: 600; }}
            .top-nav a:hover {{ text-decoration: underline; }}
            h1 {{ margin-bottom: 0.5rem; }}
            h2 {{ margin-top: 2rem; }}
            .details {{ color: #666; }}
            .metadata-links {{ margin-bottom: 1rem; }}
            .metadata-links a {{ color: #0366d6; text-decoration: none; }}
            .metadata-links a:hover {{ text-decoration: underline; }}
            .summary-box {{ border: 1px solid #c8d7f3; background: #f4f8ff; padding: 1rem 1.25rem; border-radius: 8px; margin: 1.25rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
            .summary-box h3 {{ margin-top: 0; margin-bottom: 0.75rem; font-size: 1.1rem; }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }}
            .summary-card {{ flex: 1 1 calc(33.333% - 1rem); min-width: 220px; background: #fff; border: 1px solid #dbe5fb; border-radius: 6px; padding: 0.75rem 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
            .summary-card h4 {{ margin: 0 0 0.4rem 0; font-size: 1rem; color: #1f3c88; }}
            .summary-card p {{ margin: 0.2rem 0; font-size: 0.95rem; }}
            .summary-card .metric {{ font-weight: 600; color: #0b5394; }}
            .table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
            .table th, .table td {{ border: 1px solid #ccc; padding: 0.4rem; text-align: left; }}
            .table-striped tbody tr:nth-child(odd) {{ background-color: #f9f9f9; }}
            .table-wrapper caption {{ font-weight: bold; margin-bottom: 0.5rem; text-align: left; }}
            .note {{ font-size: 0.85rem; color: #666; }}
            .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
            .figure-card {{ border: 1px solid #ddd; padding: 1rem; border-radius: 6px; background: #fafafa; display: flex; flex-direction: column; gap: 0.5rem; }}
            .figure-card img {{ width: 100%; height: auto; border: 1px solid #ddd; cursor: zoom-in; }}
            .figure-caption {{ font-weight: 600; }}
            .expand-btn {{ align-self: flex-start; padding: 0.35rem 0.75rem; border: 1px solid #0366d6; background: #0366d6; color: #fff; border-radius: 4px; cursor: pointer; }}
            .expand-btn:hover {{ background: #024f9b; }}
            .collapsible {{ margin-bottom: 1.5rem; border: 1px solid #ddd; border-radius: 6px; padding: 0.75rem 1rem; background: #fbfbfb; }}
            .collapsible summary {{ cursor: pointer; font-weight: 600; }}
            .collapsible[open] {{ background: #f2f7ff; }}
            .download-link {{ margin: 0.75rem 0; }}
            .download-link a {{ color: #0366d6; text-decoration: none; }}
            .download-link a:hover {{ text-decoration: underline; }}
            pre {{ background: #f4f4f4; padding: 1rem; overflow-x: auto; }}
            .modal {{ position: fixed; inset: 0; background: rgba(0,0,0,0.75); display: none; align-items: center; justify-content: center; z-index: 999; }}
            .modal.active {{ display: flex; }}
            .modal-content {{ position: relative; max-width: 90vw; max-height: 90vh; }}
            .modal-content img {{ width: 100%; height: auto; max-height: 90vh; border-radius: 6px; }}
            .modal-caption {{ margin-top: 0.5rem; text-align: center; color: #fff; }}
            .modal-close {{ position: absolute; top: -12px; right: -12px; border: none; background: #fff; border-radius: 50%; width: 32px; height: 32px; font-size: 1.2rem; cursor: pointer; }}
          </style>
        </head>
        <body>
          <h1>Per-Class Trimodal vs Bimodal Analysis</h1>
          {build_nav()}
          {summary_section}
          {figures_section}
          {retrieval_figures_html}
          {tables_section}
          {retrieval_tables_section}
          {json_section}
          <div id="modal" class="modal" role="dialog" aria-modal="true" aria-hidden="true">
            <div class="modal-content">
              <button type="button" class="modal-close" aria-label="Close">&times;</button>
              <img src="" alt="Expanded figure" />
              <div class="modal-caption"></div>
            </div>
          </div>
          <script>
          document.addEventListener('DOMContentLoaded', function() {{
            const modal = document.getElementById('modal');
            const modalImg = modal.querySelector('img');
            const modalCaption = modal.querySelector('.modal-caption');
            const closeBtn = modal.querySelector('.modal-close');

            function openModal(src, title) {{
              modalImg.src = src;
              modalCaption.textContent = title || '';
              modal.classList.add('active');
              modal.setAttribute('aria-hidden', 'false');
            }}

            function closeModal() {{
              modal.classList.remove('active');
              modalImg.src = '';
              modalCaption.textContent = '';
              modal.setAttribute('aria-hidden', 'true');
            }}

            document.body.addEventListener('click', function(event) {{
              const button = event.target.closest('.expand-btn');
              const image = event.target.closest('.figure-card')?.querySelector('img');
              if (button) {{
                openModal(button.dataset.full, button.dataset.title);
              }} else if (image && event.target === image) {{
                openModal(image.dataset.full, image.dataset.title);
              }} else if (event.target === modal) {{
                closeModal();
              }}
            }});

            closeBtn.addEventListener('click', closeModal);
            document.addEventListener('keydown', function(event) {{
              if (event.key === 'Escape' && modal.classList.contains('active')) {{
                closeModal();
              }}
            }});
          }});
          </script>
        </body>
        </html>
        """
    )

    REPORT_PATH.write_text(html)
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
