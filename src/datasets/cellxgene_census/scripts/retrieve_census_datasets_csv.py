import cellxgene_census

# Description: Retrieve the datasets from the census and save them as a CSV file.
with cellxgene_census.open_soma(
    census_version=snakemake.params.census_version
) as census:
    census_datasets = census["census_info"]["datasets"].read().concat().to_pandas()
    census_datasets = census_datasets.set_index("dataset_id")
    census_datasets.to_csv(snakemake.output[0])
