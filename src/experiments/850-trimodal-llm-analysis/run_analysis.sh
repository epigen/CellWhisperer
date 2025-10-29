#!/bin/bash
# Quick-start script to run complete trimodal analysis pipeline

set -e  # Exit on error

echo "=========================================="
echo "Trimodal Analysis Pipeline"
echo "=========================================="
echo ""

echo "[1/3] Running per-class analysis..."
uv run python analysis/per_class_analysis.py
echo "✓ Per-class analysis complete"
echo ""

echo "[2/3] Running retrieval analysis..."
uv run python analysis/retrieval_analysis.py
echo "✓ Retrieval analysis complete"
echo ""

echo "[3/3] Generating HTML report..."
uv run python analysis/generate_report.py
echo "✓ HTML report generated"
echo ""

echo "=========================================="
echo "Analysis complete!"
echo "=========================================="
echo ""
echo "View results:"
echo "  - HTML report: analysis/outputs/per_class_summary.html"
echo "  - CSV outputs: analysis/outputs/*.csv"
echo "  - Plots: analysis/outputs/plots/*.png"
echo "  - Retrieval data: analysis/outputs/retrieval/"
echo ""

