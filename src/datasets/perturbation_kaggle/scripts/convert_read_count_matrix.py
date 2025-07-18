import pandas as pd
import numpy as np
import anndata
from tqdm import tqdm
from pathlib import Path


xlist = ["cell_type", "sm_name"]
_ylist = ["sm_lincs_id", "SMILES", "control"]
de = pd.read_parquet(snakemake.input["de"]).set_index(xlist).drop(columns=_ylist)
adata = pd.read_parquet(snakemake.input["adata"])
meta = pd.read_csv(snakemake.input["meta"])

if not snakemake.params.extensive:
    adata = adata[adata["gene"].isin(de.columns)]

# Get unique obs_id and gene values
obs_ids = adata["obs_id"].unique()

genes = adata["gene"].unique()

# Create a dictionary to map obs_ids and genes to indices
obs_id_dict = {id: i for i, id in enumerate(obs_ids)}
gene_dict = {gene: i for i, gene in enumerate(genes)}

# Create an empty numpy array with obs_ids as rows and genes as columns
adata_array = np.zeros((len(obs_ids), len(genes)), dtype="int16")

# Fill the numpy array in chunks
chunksize = 100000  # adjust this value to fit your memory constraints
for start in tqdm(range(0, len(adata), chunksize)):  # use len(adata)
    end = start + chunksize
    chunk = adata.iloc[start:end]
    for idx, row in chunk.iterrows():
        i = obs_id_dict[row["obs_id"]]
        j = gene_dict[row["gene"]]
        adata_array[i, j] = row["count"]

# Add annotations
if snakemake.params.extensive:
    raise NotImplementedError
    # library_id - A unique identifier for each library, which is a measurement made on pooled samples from each row of the plate. All cells from wells on the same row of the same plate will share a library_id.
    # plate_name - A unique ID for all samples from the same plate.
    # well - The well location of the sample on each plate (this is standard across 96 well plate experiments). It is a concatenation of row and col.
    # row - Which row on the plate the sample came from.
    # col - Which column on the plate the sample came from.
    # donor_id - Identifies the donor source of the sample, one of three.
    # cell_type - The annotated cell type of each cell based on RNA expression. This matches the cell_type in the de_train.parquet.
    # cell_id - This is included for consistency with LINCS Connectivity Map metadata, which denotes a cell_id for each cell line.
    # sm_name - The primary name for the (parent) compound (in a standardized representation) as chosen by LINCS. This is provided to map the data in this experiment to the LINCS Connectivity Map data.
    # sm_lincs_id - The global LINCS ID (parent) compound (in a standardized representation). This is provided to map the data in this experiment to the LINCS Connectivity Map data.
    # SMILES - Simplified molecular-input line-entry system (SMILES) representations of the compounds used in the experiment. This is a 1D representation of molecular structure. These SMILES are provided by Cellarity based on the specific compounds ordered for this experiment.
    # dose_uM - Dose of the compound in on a micro-molar scale. This maps to the pert_idose field in LINCS.
    # timepoint_hr - Duration of treatment in hours. This maps to the pert_itime field in LINCS.
    # control - Whether this observation was used as a control, True or False.
else:
    pass

# Add the missing columns to de (with 0 values). make sure all genes are columns
de.index = de.index.map(lambda idx: f"{idx[0]}_{idx[1]}")
de = de.reindex(columns=genes, fill_value=0)

# Create the anndata object
dataset = anndata.AnnData(
    X=adata_array, obs=meta.set_index("obs_id").loc[obs_ids], var=de.T
)

# Two columns of the plates were dedicated to positive controls (dabrfenib and belinostat) and one column was dedicated to a negative control (DMSO="Dimethyl Sulfoxide").
dataset.obs[snakemake.params.anndata_text_obs_label] = dataset.obs.apply(
    (
        lambda row: snakemake.params.embedding_sentence.format(
            cell_type=row["cell_type"],
            sm_name=row["sm_name"],
            sm_lincs_id=row["sm_lincs_id"],
        )
    ),
    axis=1,
)

# Save dataset
Path(snakemake.output[0]).parent.mkdir(exist_ok=True, parents=True)
dataset.write(snakemake.output[0])
