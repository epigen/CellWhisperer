#!/bin/bash
# Run IHC correlation analysis
#
# Usage:
#   bash run_ihc_correlation_analysis.sh <predictions_h5ad_file>
#
# Example:
#   bash run_ihc_correlation_analysis.sh results/predictions/TMA1_predictions.h5ad

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Error: Missing predictions file"
    echo "Usage: bash run_ihc_correlation_analysis.sh <predictions_h5ad_file>"
    exit 1
fi

PREDICTIONS_FILE=$1
MAPPING_FILE="tma_grid_to_patient_mapping.csv"
IHC_FILE="patient_ihc.xlsx"
OUTPUT_DIR="analysis"

echo "================================"
echo "IHC Correlation Analysis"
echo "================================"
echo "Predictions: $PREDICTIONS_FILE"
echo "Mapping: $MAPPING_FILE"
echo "IHC file: $IHC_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if files exist
if [ ! -f "$PREDICTIONS_FILE" ]; then
    echo "Error: Predictions file not found: $PREDICTIONS_FILE"
    exit 1
fi

if [ ! -f "$MAPPING_FILE" ]; then
    echo "Creating TMA grid to patient mapping..."
    pixi run --no-progress python << 'EOF'
import pandas as pd

# Load IHC file
ihc_df = pd.read_excel('patient_ihc.xlsx', header=1)
ihc_df['sample_id'] = ihc_df['core_id'].str.extract(r'-0*(\d+)$')[0]

# Load mapping from dataset
mapping_df = pd.read_csv('../../../src/datasets/lymphoma_cosmx_large/cell_barcode_core_assignment.csv')
grid_to_sample = mapping_df[['core_id', 'sample_id']].drop_duplicates().rename(columns={'core_id': 'grid_position'})

# Merge to create complete mapping
complete_map = pd.merge(
    grid_to_sample,
    ihc_df[['Blocks', 'core_id', 'sample_id', 'PAX5 H-score', 'CD19']],
    on='sample_id',
    how='inner'
)

# Save mapping
complete_map.to_csv('tma_grid_to_patient_mapping.csv', index=False)
print(f"Created mapping with {len(complete_map)} cores")
EOF
fi

# Run correlation analysis
echo "Running correlation analysis..."
pixi run --no-progress python scripts/correlate_predictions_with_ihc.py \
    --predictions "$PREDICTIONS_FILE" \
    --mapping "$MAPPING_FILE" \
    --ihc "$IHC_FILE" \
    --output "$OUTPUT_DIR"

echo ""
echo "================================"
echo "Analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "================================"
