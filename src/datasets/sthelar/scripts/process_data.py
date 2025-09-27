"""
Process downloaded STHELAR dataset for CellWhisperer.

This script processes the downloaded STHELAR zarr files and H&E patches
to be compatible with the existing processing pipeline requirements.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
import zipfile
import tempfile
import shutil
from pathlib import Path
import logging
import anndata as ad
from PIL import Image
import geopandas as gpd
from shapely.geometry import Point, Polygon
import zarr
import h5py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sthelar_slide(zarr_path):
    """Load a STHELAR slide from zarr file."""
    logger.info(f"Loading zarr file: {zarr_path}")
    
    # Extract zarr file to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        with zipfile.ZipFile(zarr_path, 'r') as zip_file:
            zip_file.extractall(temp_path)
        
        # Find the extracted zarr directory
        zarr_dirs = list(temp_path.glob("*.zarr"))
        if not zarr_dirs:
            raise ValueError(f"No zarr directory found in {zarr_path}")
        
        zarr_dir = zarr_dirs[0]
        
        # Load the SpatialData object
        try:
            sdata = sd.read_zarr(str(zarr_dir))
        except ImportError:
            logger.warning("SpatialData not available, falling back to direct zarr loading")
            # Fallback: try to load data directly from zarr structure
            import zarr
            zarr_store = zarr.open(str(zarr_dir), mode='r')
            sdata = zarr_store
        
    return sdata


def extract_patches_from_images_zip(images_zip_path, sample_id, output_dir):
    """Extract H&E patches for a specific slide from images.zip."""
    logger.info(f"Extracting patches for sample {sample_id}")
    
    patches = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(images_zip_path, 'r') as zip_file:
        # List all files in the zip
        file_list = zip_file.namelist()
        
        # Filter files for this slide (format: slide-id_patch-id.*)
        slide_files = [f for f in file_list if f.startswith(f"{sample_id}_")]
        
        logger.info(f"Found {len(slide_files)} patch files for sample {sample_id}")
        
        for file_path in slide_files:
            # Extract patch ID from filename
            filename = Path(file_path).name
            if "_" in filename:
                patch_id = filename.split("_", 1)[1].split(".")[0]
                
                # Extract the image file
                with zip_file.open(file_path) as src_file:
                    image_data = src_file.read()
                    
                # Save to temporary file and load as image
                temp_file = output_dir / f"temp_{patch_id}.png"
                with open(temp_file, 'wb') as f:
                    f.write(image_data)
                
                # Load image as numpy array
                image = Image.open(temp_file)
                patches[patch_id] = np.array(image)
                
                # Clean up temp file
                temp_file.unlink()
    
    return patches


def map_cells_to_patches(sdata, patch_coords):
    """Map cells to patches based on their spatial coordinates."""
    logger.info("Mapping cells to patches")
    
    # Get cell boundaries
    if 'cell_boundaries' not in sdata.shapes:
        raise ValueError("No cell_boundaries found in SpatialData")
    
    cell_boundaries = sdata.shapes['cell_boundaries']
    
    # Get patch coordinates
    if 'he_patches' not in sdata.shapes:
        raise ValueError("No he_patches found in SpatialData")
    
    he_patches = sdata.shapes['he_patches']
    
    cell_to_patch = {}
    
    # For each patch, find overlapping cells
    for patch_idx, patch_row in he_patches.iterrows():
        patch_geometry = patch_row.geometry
        
        # Find cells that intersect with this patch
        overlapping_cells = []
        for cell_idx, cell_row in cell_boundaries.iterrows():
            cell_geometry = cell_row.geometry
            
            if patch_geometry.intersects(cell_geometry):
                overlapping_cells.append(cell_idx)
        
        cell_to_patch[patch_idx] = overlapping_cells
    
    return cell_to_patch


def aggregate_gene_expression_to_patches(sdata, cell_to_patch_mapping):
    """Aggregate single-cell gene expression to patch level."""
    logger.info("Aggregating gene expression to patch level")
    
    # Get the cell expression table
    if 'table_cells' not in sdata.tables:
        raise ValueError("No table_cells found in SpatialData")
    
    cell_table = sdata.tables['table_cells']
    
    # Create patch-level expression matrix
    patch_expressions = {}
    patch_cell_counts = {}
    
    for patch_id, cell_ids in cell_to_patch_mapping.items():
        if len(cell_ids) == 0:
            continue
            
        # Get expression data for cells in this patch
        try:
            patch_cells = cell_table[cell_table.obs.index.isin(cell_ids)]
            
            if len(patch_cells) > 0:
                # Sum expression across cells in the patch
                patch_expression = np.array(patch_cells.X.sum(axis=0)).flatten()
                patch_expressions[patch_id] = patch_expression
                patch_cell_counts[patch_id] = len(patch_cells)
            
        except Exception as e:
            logger.warning(f"Failed to process patch {patch_id}: {e}")
            continue
    
    return patch_expressions, patch_cell_counts


def create_adata_for_slide(sdata, patches, patch_expressions, patch_cell_counts, sample_id):
    """Create AnnData object for a slide with patch-level data."""
    logger.info(f"Creating AnnData for sample {sample_id}")
    
    # Get gene names from the cell table
    cell_table = sdata.tables['table_cells']
    gene_names = cell_table.var.index.tolist()
    
    # Create expression matrix for patches
    patch_ids = list(patch_expressions.keys())
    
    if len(patch_ids) == 0:
        raise ValueError(f"No patches with expression data found for sample {sample_id}")
    
    # Create expression matrix (patches x genes)
    expression_matrix = np.zeros((len(patch_ids), len(gene_names)))
    
    patch_obs_data = []
    
    for i, patch_id in enumerate(patch_ids):
        expression_matrix[i, :] = patch_expressions[patch_id]
        
        # Create observation metadata
        patch_obs = {
            'patch_id': patch_id,
            'slide_id': sample_id,
            'cell_count': patch_cell_counts.get(patch_id, 0),
            # Add spatial coordinates if available
            'x_pixel': 0,  # Will be updated below if patch coordinates available
            'y_pixel': 0
        }
        patch_obs_data.append(patch_obs)
    
    # Try to get patch coordinates
    if 'he_patches' in sdata.shapes:
        he_patches = sdata.shapes['he_patches']
        for i, patch_id in enumerate(patch_ids):
            if patch_id in he_patches.index:
                patch_geom = he_patches.loc[patch_id].geometry
                if hasattr(patch_geom, 'centroid'):
                    centroid = patch_geom.centroid
                    patch_obs_data[i]['x_pixel'] = int(centroid.x)
                    patch_obs_data[i]['y_pixel'] = int(centroid.y)
    
    # Create AnnData object
    adata = ad.AnnData(
        X=expression_matrix,
        obs=pd.DataFrame(patch_obs_data, index=patch_ids),
        var=pd.DataFrame(index=gene_names)
    )
    
    # Add gene names to var
    adata.var['gene_name'] = gene_names
    
    # Add counts layer (assuming X contains raw counts)
    adata.layers['counts'] = adata.X.copy()
    
    # Add slide image if we have patch data
    if patches and len(patches) > 0:
        # Create a mosaic of patches or use first patch as representative
        first_patch_id = list(patches.keys())[0]
        first_patch = patches[first_patch_id]
        
        # Use first patch as representative slide image
        # In a more sophisticated approach, we could create a mosaic
        adata.uns['20x_slide'] = first_patch
        
        # Set spot diameter for patch extraction
        adata.uns['spot_diameter_fullres'] = 256  # typical patch size
    
    # Add sample metadata
    adata.uns['sample_id'] = sample_id
    adata.uns['dataset'] = 'sthelar'
    
    return adata


def prepare_adata_for_uniprocessor(adata):
    """Prepare AnnData to match UNIProcessor requirements (similar to HEST1K)."""
    
    # Ensure gene names are uppercase
    adata.var.index = adata.var.index.astype(str).str.upper()
    
    if 'gene_name' not in adata.var.columns:
        adata.var['gene_name'] = adata.var.index
    
    # Ensure counts layer exists
    if 'counts' not in adata.layers:
        adata.layers['counts'] = adata.X.astype(int)
    
    # Ensure spatial coordinates are integers
    if 'x_pixel' in adata.obs:
        adata.obs['x_pixel'] = adata.obs['x_pixel'].astype(int)
    if 'y_pixel' in adata.obs:
        adata.obs['y_pixel'] = adata.obs['y_pixel'].astype(int)
    
    return adata


# Get parameters from snakemake
sample_id = snakemake.params.sample_id
cache_dir = Path(snakemake.params.sthelar_cache_dir)

logger.info(f"Processing sample {sample_id}")

# Paths to downloaded data
zarr_path = cache_dir / f"sdata_{sample_id}.zarr.zip"
images_zip_path = cache_dir / "images.zip"

if not zarr_path.exists():
    raise FileNotFoundError(f"Zarr file not found: {zarr_path}")
if not images_zip_path.exists():
    raise FileNotFoundError(f"Images zip not found: {images_zip_path}")

# Load the SpatialData object
sdata = load_sthelar_slide(zarr_path)

# Extract H&E patches for this slide
with tempfile.TemporaryDirectory() as temp_dir:
    patches = extract_patches_from_images_zip(images_zip_path, sample_id, temp_dir)

# Map cells to patches
cell_to_patch_mapping = map_cells_to_patches(sdata, patches)

# Aggregate gene expression to patch level
patch_expressions, patch_cell_counts = aggregate_gene_expression_to_patches(
    sdata, cell_to_patch_mapping
)

# Create AnnData object
adata = create_adata_for_slide(
    sdata, patches, patch_expressions, patch_cell_counts, sample_id
)

# Prepare for UNIProcessor compatibility
adata = prepare_adata_for_uniprocessor(adata)

# Save the processed data
logger.info(f"Saving processed data to {snakemake.output.full_data_file}")
adata.write_h5ad(snakemake.output.full_data_file)

logger.info(f"Successfully processed sample {sample_id}")