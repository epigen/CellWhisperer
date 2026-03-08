# UCE Gene Data Files

These files are required for the new UCE model (Kuan's `data_collection_exp` implementation).

## Files

- `gene_names.txt` — List of 5,782 gene names (CosMx 6K + cellxgene_census overlap). Specifies which genes from h5ad to include.
- `all_species_gene_dict.json` — Gene mapping dictionary keyed by species (e.g. "human"). Maps gene names to protein embedding IDs, chromosome IDs, and genomic locations. Contains 19,656 genes; 5,762 overlap with gene_names.txt (20 genes in gene_names.txt are not in the mapping).

## Provenance

**gene_names.txt**: 
- Original: `/home/moritz/Downloads/cosmx6k_cxg_overlap_genes.csv`
- 5,782 genes that are the intersection of CosMx 6K panel and cellxgene_census

**all_species_gene_dict.json**: 
- Originally downloaded from: https://drive.google.com/drive/folders/1pWHRSqm3Njz1KPRfMP-ISNts785I2COG
- Copied from: `/home/moritz/code/pert_lang_dataset/resources/`

## Model Checkpoint

The HuggingFace model checkpoint `KuanP/uce-cosmx-geneset` is auto-downloaded at runtime.
This model was trained on the CosMx gene set.
