import os
import shutil
import anndata as ad
import sys
import pandas as pd

def set_default_spatial_embedding():
    """
    Iterates through specified AnnData files, backs them up,
    sanitizes data types to prevent h5py errors, and sets 'X_spatial' 
    as the default embedding in adata.uns.
    """
    base_path = "/nobackup/lab_bock/projects/cellwhisperer/Jake"
    embedding_key_to_set = 'X_spatial'

    print("--- Starting script to set default embedding ---")

    for i in range(1, 6):
        h5ad_dir = os.path.join(base_path, f"lc_{i}", "cellwhisperer_clip_v1")
        h5ad_path = os.path.join(h5ad_dir, "cellxgene.h5ad")
        backup_path = os.path.join(h5ad_dir, "cellxgene.h5ad.bak")

        print(f"\n--- Processing: {h5ad_path} ---")

        if not os.path.exists(h5ad_path):
            print(f"⚠️  WARNING: File not found. Skipping.")
            continue

        try:
            print(f"Backing up to: {backup_path}")
            shutil.copy2(h5ad_path, backup_path)
            print("Backup successful.")

            print("Loading AnnData file...")
            adata = ad.read_h5ad(h5ad_path)

            if embedding_key_to_set not in adata.obsm:
                print(f"❌ ERROR: Embedding '{embedding_key_to_set}' not found in .obsm.")
                print(f"Available keys are: {list(adata.obsm.keys())}")
                continue
            
            print(f"Setting 'default_embedding' to '{embedding_key_to_set}'...")
            adata.uns['default_embedding'] = embedding_key_to_set
            
            """
            # --- NEW: Sanitize DataFrame dtypes before saving ---
            print("Sanitizing dtypes to prevent h5py error...")
            for df_name in ['obs', 'var']:
                df = getattr(adata, df_name)
                for col in df.columns:
                    if pd.api.types.is_categorical_dtype(df[col]):
                        # Convert categories to strings if they are not already
                        if not all(isinstance(c, str) for c in df[col].cat.categories):
                            print(f"Converting categorical column '{col}' in 'adata.{df_name}' to string type.")
                            df[col] = df[col].astype(str)

			"""
            print(f"Saving changes to {h5ad_path}...")
            adata.write_h5ad(h5ad_path)
            print("✅ Successfully updated file.")

        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            print(f"This could be the h5py precision error. If you haven't, try 'pip install h5py==2.10.0'")
            continue

    print("\n--- Script finished ---")


# --- Run the function ---
# First, ensure you have the necessary libraries installed
try:
    import anndata
    import pandas
except ImportError as e:
    print(f"Missing required library: {e.name}. Please install it by running 'pip install {e.name}'")
    sys.exit(1)

set_default_spatial_embedding()