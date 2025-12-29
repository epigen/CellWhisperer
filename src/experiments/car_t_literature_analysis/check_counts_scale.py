import warnings
warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
from scipy.sparse import issparse

H5AD_PATH = "cellxgene_B_Product_lowburden.light.h5ad"

adata = ad.read_h5ad(H5AD_PATH)

# Prefer explicit counts layer if present, else X
if "counts" in adata.layers:
    counts = adata.layers["counts"]
    source = "layers['counts']"
else:
    counts = adata.X
    source = "X"

# Extract nonzero values for statistics
if issparse(counts):
    nonzero = counts.data
    max_val = float(counts.max())
    nnz = counts.nnz
    n = counts.shape[0] * counts.shape[1]
else:
    arr = counts.A if hasattr(counts, "A") else counts
    nonzero = arr[arr > 0]
    max_val = float(arr.max())
    nnz = int((arr > 0).sum())
    n = arr.size

# Handle empty edge case gracefully
if nonzero.size == 0:
    print(f"Source: {source}\nNo nonzero entries found.")
else:
    q95 = float(np.quantile(nonzero, 0.95))
    q99 = float(np.quantile(nonzero, 0.99))
    mean_nz = float(nonzero.mean())
    frac_integer = float(np.mean(np.isclose(nonzero, np.round(nonzero))))
    frac_lt_20 = float(np.mean(nonzero < 20.0))

    print(
        "\n".join(
            [
                f"Source: {source}",
                f"Total entries: {n:,} | Nonzero: {nnz:,}",
                f"Max value: {max_val:.4f}",
                f"Mean(nonzero): {mean_nz:.4f}",
                f"95th pct(nonzero): {q95:.4f}",
                f"99th pct(nonzero): {q99:.4f}",
                f"Fraction(nonzero) < 20: {frac_lt_20:.4f}",
                f"Fraction(nonzero) integer-like: {frac_integer:.4f}",
                f"Heuristic: 99th<20 -> likely log1p: {q99 < 20.0}",
            ]
        )
    )
