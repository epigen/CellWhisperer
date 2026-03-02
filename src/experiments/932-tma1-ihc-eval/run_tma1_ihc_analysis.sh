#!/bin/bash
# Run TMA1 IHC correlation analysis
#
# This script analyzes TMA1 predictions and correlates with IHC ground truth.
# It uses the TMA1-specific grid position mapping.

set -e

echo "================================"
echo "TMA1 IHC Correlation Analysis"
echo "================================"

# Configuration
MAPPING_FILE="tma1_fov_to_ihc_mapping.csv"
IHC_FILE="patient_ihc.xlsx"
OUTPUT_DIR="analysis"

# Check if mapping file exists, if not create it
if [ ! -f "$MAPPING_FILE" ]; then
    echo "Creating TMA1 FOV to IHC mapping..."
    pixi run --no-progress python << 'EOF'
import pandas as pd

# Load TMA1 mapping with FOV
tma1_mapping = pd.read_csv('../../datasets/lymphoma_cosmx_small/cell_barcode_core_assignment.csv')

# Load IHC file
ihc_df = pd.read_excel('patient_ihc.xlsx', header=1)
ihc_df['sample_id'] = ihc_df['core_id'].str.extract(r'-0*(\d+)$')[0]

# Create FOV to grid position mapping
fov_to_grid = tma1_mapping[['fov', 'core_id', 'sample_id']].drop_duplicates()
fov_to_grid = fov_to_grid.rename(columns={'core_id': 'grid_position'})

# Merge with IHC
complete_map = pd.merge(
    fov_to_grid,
    ihc_df[['sample_id', 'Blocks', 'core_id', 'PAX5 H-score', 'CD19']],
    on='sample_id',
    how='inner'
)

complete_map.to_csv('tma1_fov_to_ihc_mapping.csv', index=False)
print(f"✓ Created mapping with {len(complete_map)} FOVs")
print(f"✓ {len(complete_map.dropna(subset=['PAX5 H-score', 'CD19']))} FOVs have both PAX5 and CD19")
EOF
fi

# Find predictions file
echo ""
echo "Looking for TMA1 predictions file..."

# Check possible locations
PREDICTIONS_FILE=""

if [ -f "results/predictions/TMA1_predictions.h5ad" ]; then
    PREDICTIONS_FILE="results/predictions/TMA1_predictions.h5ad"
elif [ -f "../../../results/experiments/932-tma1-ihc-eval/predictions/TMA1_predictions.h5ad" ]; then
    PREDICTIONS_FILE="../../../results/experiments/932-tma1-ihc-eval/predictions/TMA1_predictions.h5ad"
elif [ -f "../../../results/lymphoma_cosmx_large/h5ads/full_data_TMA1.h5ad" ]; then
    echo "WARNING: Using source TMA1 data (no predictions found)"
    echo "This will show ground truth correlations, not predictions!"
    PREDICTIONS_FILE="../../../results/lymphoma_cosmx_large/h5ads/full_data_TMA1.h5ad"
elif [ -f "../core_tma_alignment/TMA1.h5ad" ]; then
    echo "WARNING: Using TMA1 alignment data (no predictions found)"
    echo "This will show ground truth correlations, not predictions!"
    PREDICTIONS_FILE="../core_tma_alignment/TMA1.h5ad"
else
    echo "ERROR: Could not find TMA1 predictions file!"
    echo ""
    echo "Checked locations:"
    echo "  - results/predictions/TMA1_predictions.h5ad"
    echo "  - ../../../results/experiments/932-tma1-ihc-eval/predictions/TMA1_predictions.h5ad"
    echo "  - ../../../results/lymphoma_cosmx_large/h5ads/full_data_TMA1.h5ad"
    echo "  - ../core_tma_alignment/TMA1.h5ad"
    echo ""
    echo "Please provide the path to TMA1 predictions:"
    echo "  bash $0 <path_to_TMA1_predictions.h5ad>"
    exit 1
fi

# Allow override via command line argument
if [ $# -ge 1 ]; then
    PREDICTIONS_FILE=$1
fi

echo "Using predictions file: $PREDICTIONS_FILE"
echo ""

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
echo "================================"
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.csv "$OUTPUT_DIR"/*.png 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
