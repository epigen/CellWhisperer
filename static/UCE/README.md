# UCE Gene Data Files

These files are required for the new UCE model (Kuan's `data_collection_exp` implementation).

## Files

- `gene_names.txt` — List of gene names (vocabulary). Specifies which genes from h5ad to include.
- `all_species_gene_dict.json` — Gene mapping dictionary keyed by species (e.g. "human"). Maps gene names to protein embedding IDs, chromosome IDs, and genomic locations.

## Provenance

Originally downloaded from:
https://drive.google.com/drive/folders/1pWHRSqm3Njz1KPRfMP-ISNts785I2COG

Copied from: `/home/moritz/code/pert_lang_dataset/resources/`

See also: `modules/data_collection_exp/README.md` for more context.

## Model Checkpoint

The HuggingFace model checkpoint `KuanP/uce-cosmx-geneset` is auto-downloaded at runtime.
