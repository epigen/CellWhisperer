# Dataset processing

For efficient use of CellWhisperer in the web browser (CELLxGENE Explorer integration), you need to preprocess your datasets. Note that this is a much more efficient process, if you use a GPU.

To do so, follow these simple steps:

1. Prepare your dataset (for guidelines see below)
2. Place it in `<PROJECT_ROOT>/resources/<dataset_name>/read_count_table.h5ad`
3. Run this pipeline: `snakemake --config 'datasets=["<dataset_name>"]'`

The final file will be generated here: `<PROJECT_ROOT>/results/<dataset_name>/cellwhisperer_clip_v1/cellxgene.h5ad`

## Dataset input format guidelines

We only use human data and raw read counts (not normalized) for our datasets. Normalization is taken care of by the respective transcriptome models (more specifically their processor classes) and is also performed explicitly in this preparation pipeline.

- A dataset is stored in an h5ad file
- `X` contains raw read counts and without nans (use int32)
- `var` has a *unique* index (e.g. the ensembl_id (not mandatory, but recommended)) and an additional field `gene_name` containing the gene symbol.
  - Optionally, provide an additional field "ensembl_id" (otherwise the pipeline computes it).
- If your dataset is large (i.e. > 100k cells), restrict the provided metadata fields (e.g. in `obs` and `var`) to what is really necessary
- For best results, filter cells with few expressed genes (e.g. <100 genes with expression <1)
- Try to use `categorical` instead of 'object' dtype for categorical `obs` columns
- Any layouts that should make it into the webapp need to adhere to these rules:
  - stored in `.obsm` whith name `X_{name}`
  - type: `np.ndarray` (NOT `pd.DataFrame`), dtype: float/int/uint
  - shape: `(n_obs, >= 2)`
  - all values finite or NaN (NO +Inf or -Inf)
