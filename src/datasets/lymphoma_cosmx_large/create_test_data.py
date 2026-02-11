import numpy as np
import pandas as pd
import anndata
from PIL import Image
import os
from pathlib import Path

# Check if required packages are available
try:
    import numpy as np
    import pandas as pd
    import anndata
    from PIL import Image
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages with:")
    print("pip install anndata pandas pillow numpy")
    exit(1)

# Create test directories
test_dir = Path("resources/lymphoma_cosmx_small")
test_dir.mkdir(parents=True, exist_ok=True)

# 1. Create a mock SVS image (we'll create a simple TIFF that can be opened by openslide)
# Create a 1000x1000 RGB image with some random patterns
np.random.seed(42)
image_size = 1000
image_array = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)

# Add some structure - make some areas whiter (background) and some darker (tissue)
# Create tissue-like regions
for i in range(5):
    center_x = np.random.randint(100, 900)
    center_y = np.random.randint(100, 900)
    radius = np.random.randint(50, 150)

    y, x = np.ogrid[:image_size, :image_size]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2

    # Make tissue regions darker (more colorful)
    image_array[mask] = np.random.randint(50, 200, (np.sum(mask), 3))

# Make background regions whiter
background_mask = np.random.random((image_size, image_size)) > 0.3
image_array[background_mask] = np.random.randint(200, 255, (np.sum(background_mask), 3))

# Save as TIFF (openslide can read TIFF files)
image = Image.fromarray(image_array)
image_path = test_dir / "image.svs"
# Save directly as .svs (which is essentially a TIFF)
image.save(str(image_path), format="TIFF")

print(f"Created mock SVS image at {image_path}")

# 2. Create mock AnnData with spatial coordinates and expression data
n_cells = 2000
n_genes = 500

# Generate random gene names
gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]

# Generate random expression data (sparse-like, mostly zeros with some high values)
np.random.seed(42)
expression_data = np.random.negative_binomial(
    n=5, p=0.8, size=(n_cells, n_genes)
).astype(float)

# Generate spatial coordinates that fall within our image bounds
# Make sure some cells are in tissue regions and some in background
spatial_coords = np.random.uniform(0, image_size, (n_cells, 2))

# Create some clustering of cells (more realistic)
n_clusters = 8
cluster_centers = np.random.uniform(100, 900, (n_clusters, 2))
cluster_assignments = np.random.randint(0, n_clusters, n_cells)

for i in range(n_cells):
    cluster_id = cluster_assignments[i]
    # Add some noise around cluster centers
    noise = np.random.normal(0, 30, 2)
    spatial_coords[i] = cluster_centers[cluster_id] + noise

    # Ensure coordinates stay within bounds
    spatial_coords[i] = np.clip(spatial_coords[i], 0, image_size - 1)

# Create cell metadata
cell_ids = [f"Cell_{i:06d}" for i in range(n_cells)]
cell_types = np.random.choice(
    ["T_cell", "B_cell", "Macrophage", "NK_cell", "Epithelial"], n_cells
)

# Convert pixel coordinates back to mm coordinates for testing
# This reverses the transformation that will be applied in process_data.py
# For test purposes, we'll create reasonable mm coordinates
x_slide_mm = np.random.uniform(-1.0, 8.0, n_cells)  # Reasonable range in mm
y_slide_mm = np.random.uniform(-1.5, 8.0, n_cells)  # Reasonable range in mm

# Create AnnData object
adata = anndata.AnnData(
    X=expression_data,
    obs=pd.DataFrame(
        {
            "cell_id": cell_ids,
            "cell_type": cell_types,
            "cluster": cluster_assignments,
            "x_slide_mm": x_slide_mm,
            "y_slide_mm": y_slide_mm,
        },
        index=cell_ids,
    ),
    var=pd.DataFrame(
        {
            "gene_name": gene_names,
            "highly_variable": np.random.choice([True, False], n_genes, p=[0.2, 0.8]),
        },
        index=gene_names,
    ),
)

# Add spatial coordinates to obsm
adata.obsm["spatial"] = spatial_coords

# Add some additional metadata
adata.obs["total_counts"] = np.array(adata.X.sum(axis=1)).flatten()
adata.obs["n_genes"] = np.array((adata.X > 0).sum(axis=1)).flatten()

# Save the AnnData object
adata_path = test_dir / "read_count_table.h5ad"
adata.write_h5ad(adata_path)

print(f"Created mock AnnData with {n_cells} cells and {n_genes} genes at {adata_path}")
print(
    f"Spatial coordinates range: X=[{spatial_coords[:, 0].min():.1f}, {spatial_coords[:, 0].max():.1f}], Y=[{spatial_coords[:, 1].min():.1f}, {spatial_coords[:, 1].max():.1f}]"
)
print(f"Expression data shape: {adata.X.shape}")
print(f"Cell types: {np.unique(cell_types)}")

print("\nTest data creation complete!")
print("You can now run the pipeline with:")
print("cd src/datasets/lymphoma_cosmx_small && snakemake -j1")
